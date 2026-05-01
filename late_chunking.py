"""Late-chunking with section-aware windowing for long-context encoders.

The pipeline is:

  1. Find section spans (character coordinates) in an article's rendered Markdown.
  2. Tokenize the whole article once, with offset mapping, so each section maps
     to a token range.
  3. Greedy-pack consecutive sections into windows of `core` tokens, where
     `core <= context_limit - 2 * margin`. Sections too large for one window
     get split into fragments; fragments live in their own windows.
  4. Surround each window's core with up to `margin` tokens of left- and
     right-context drawn from the surrounding tokens of the same article.
     Margins exist purely so attention can flow between adjacent sections.
  5. Run the model once per window. Mean-pool the *core* token outputs (skip
     the margin) per section. If a section was split, weighted-average its
     fragment vectors by token count to recover one vector per section.

The output is one embedding vector per source section, in article order.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# Markdown heading lines: between 1 and 6 leading hashes followed by whitespace.
HEADING_LINE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class SectionCharSpan:
    """A section's character-coordinate span and its heading metadata."""

    char_start: int
    char_end: int
    heading_level: int  # 0 for the lead section (no heading)
    heading_text: str | None  # None for the lead section


@dataclass(frozen=True, slots=True)
class SectionTokenSpan:
    """A section's token-coordinate span (after tokenization)."""

    section_index: int
    token_start: int  # inclusive, in article token coordinates
    token_end: int  # exclusive
    heading_level: int
    heading_text: str | None


@dataclass(frozen=True, slots=True)
class CoreSegment:
    """One contiguous range of tokens inside a window's core that pools to part
    of (or all of) a single section's vector.

    For a section that fits in one window, fragment_count == 1. For a section
    split across N windows, the same section_index produces N CoreSegments,
    each with the same fragment_count == N, but different fragment_index in
    [0, N)."""

    section_index: int
    fragment_index: int
    fragment_count: int
    article_token_start: int  # in article coordinates
    article_token_end: int  # exclusive


@dataclass(frozen=True, slots=True)
class Window:
    """One forward-pass through the encoder.

    `token_ids` is the literal input to the model: left margin (0..core_start),
    then core (core_start..core_end), then right margin (core_end..len).
    `core_segments` describe how to mean-pool the core token outputs into
    section fragments.
    """

    token_ids: list[int]
    core_start: int  # offset in token_ids where core begins
    core_end: int  # offset where core ends (exclusive)
    core_segments: list[CoreSegment]
    article_left_margin_start: int  # in article token coordinates
    article_right_margin_end: int  # exclusive

    @property
    def length(self) -> int:
        return len(self.token_ids)

    def core_segment_window_range(self, segment: CoreSegment) -> tuple[int, int]:
        """Return (start, end) offsets for `segment` in this window's outputs.

        A window position is just the corresponding article position shifted by
        the left-margin start, since `token_ids` is a contiguous slice of the
        article tokens beginning at `article_left_margin_start`.
        """
        start = segment.article_token_start - self.article_left_margin_start
        end = segment.article_token_end - self.article_left_margin_start
        return start, end


def find_section_char_spans(text: str) -> list[SectionCharSpan]:
    """Split the rendered Markdown text into section spans.

    A section runs from a heading line to the next heading line (exclusive).
    The lead — anything before the first heading — is its own section with
    `heading_level == 0` and `heading_text is None`.
    """
    if not text:
        return []
    headings = list(HEADING_LINE.finditer(text))
    if not headings:
        return [SectionCharSpan(0, len(text), 0, None)]

    spans: list[SectionCharSpan] = []
    if headings[0].start() > 0:
        spans.append(SectionCharSpan(0, headings[0].start(), 0, None))
    for index, match in enumerate(headings):
        next_start = (
            headings[index + 1].start() if index + 1 < len(headings) else len(text)
        )
        spans.append(
            SectionCharSpan(
                char_start=match.start(),
                char_end=next_start,
                heading_level=len(match.group(1)),
                heading_text=match.group(2).strip() or None,
            )
        )
    return spans


def section_token_spans_from_offsets(
    section_char_spans: Sequence[SectionCharSpan],
    token_offsets: Sequence[tuple[int, int]],
) -> list[SectionTokenSpan]:
    """Convert character-coordinate section spans into token-coordinate spans
    using the tokenizer's `offset_mapping` output. Sections that produce zero
    tokens (empty after tokenization) are dropped silently."""
    n_tokens = len(token_offsets)
    spans: list[SectionTokenSpan] = []
    for index, section in enumerate(section_char_spans):
        # First token whose start offset >= section.char_start.
        token_start = n_tokens
        for token_index, (char_start, _) in enumerate(token_offsets):
            if char_start >= section.char_start:
                token_start = token_index
                break
        # First token whose start offset >= section.char_end (exclusive).
        token_end = n_tokens
        for token_index, (char_start, _) in enumerate(token_offsets):
            if char_start >= section.char_end:
                token_end = token_index
                break
        if token_end > token_start:
            spans.append(
                SectionTokenSpan(
                    section_index=index,
                    token_start=token_start,
                    token_end=token_end,
                    heading_level=section.heading_level,
                    heading_text=section.heading_text,
                )
            )
    return spans


def plan_windows(
    article_token_ids: Sequence[int],
    section_token_spans: Sequence[SectionTokenSpan],
    context_limit: int,
    margin: int,
) -> list[Window]:
    """Greedy-pack sections into windows of `<= context_limit - 2*margin` core
    tokens. Sections that exceed the per-window core limit are split into
    fragments, each fragment getting its own window. Margins are added per
    window from neighboring article tokens.
    """
    if context_limit <= 2 * margin:
        raise ValueError(
            f"context_limit ({context_limit}) must exceed 2 * margin ({2 * margin})"
        )
    max_core_tokens = context_limit - 2 * margin
    n_article_tokens = len(article_token_ids)

    # Step 1: build "segments" — each segment is one fragment of one section that
    # fits in a single window. Sections shorter than max_core_tokens become a
    # single segment; longer sections become multiple segments.
    segments: list[CoreSegment] = []
    for span in section_token_spans:
        section_size = span.token_end - span.token_start
        if section_size <= max_core_tokens:
            segments.append(
                CoreSegment(
                    section_index=span.section_index,
                    fragment_index=0,
                    fragment_count=1,
                    article_token_start=span.token_start,
                    article_token_end=span.token_end,
                )
            )
            continue
        n_fragments = (section_size + max_core_tokens - 1) // max_core_tokens
        # Even-sized fragments (last one may be slightly smaller).
        base = section_size // n_fragments
        remainder = section_size - base * n_fragments
        cursor = span.token_start
        for fragment_index in range(n_fragments):
            this_size = base + (1 if fragment_index < remainder else 0)
            segments.append(
                CoreSegment(
                    section_index=span.section_index,
                    fragment_index=fragment_index,
                    fragment_count=n_fragments,
                    article_token_start=cursor,
                    article_token_end=cursor + this_size,
                )
            )
            cursor += this_size

    # Step 2: greedy-pack segments into windows. A multi-fragment section never
    # shares a window with anything else: its segments are already exactly
    # max_core_tokens (or smaller, for the last fragment), so packing them
    # alongside other sections would risk overflow.
    window_segment_groups: list[list[CoreSegment]] = []
    current: list[CoreSegment] = []
    current_size = 0
    for segment in segments:
        segment_size = segment.article_token_end - segment.article_token_start
        is_split_section = segment.fragment_count > 1
        previous_was_split = bool(current) and current[-1].fragment_count > 1
        new_window_required = (
            not current
            or is_split_section
            or previous_was_split
            or current_size + segment_size > max_core_tokens
        )
        if new_window_required and current:
            window_segment_groups.append(current)
            current = []
            current_size = 0
        current.append(segment)
        current_size += segment_size
    if current:
        window_segment_groups.append(current)

    # Step 3: materialize each window with margins drawn from the surrounding
    # article tokens.
    windows: list[Window] = []
    for group in window_segment_groups:
        core_start_in_article = group[0].article_token_start
        core_end_in_article = group[-1].article_token_end
        left_margin_start = max(0, core_start_in_article - margin)
        right_margin_end = min(n_article_tokens, core_end_in_article + margin)
        token_ids = list(article_token_ids[left_margin_start:right_margin_end])
        core_start_in_window = core_start_in_article - left_margin_start
        core_end_in_window = core_end_in_article - left_margin_start
        windows.append(
            Window(
                token_ids=token_ids,
                core_start=core_start_in_window,
                core_end=core_end_in_window,
                core_segments=list(group),
                article_left_margin_start=left_margin_start,
                article_right_margin_end=right_margin_end,
            )
        )
    return windows


def pool_section_vectors(
    windows: Sequence[Window],
    window_token_outputs: Sequence[np.ndarray],
    n_sections: int,
    embedding_dim: int,
) -> np.ndarray:
    """Reduce per-window per-token output vectors to one vector per section.

    `window_token_outputs[i]` must have shape `(windows[i].length, embedding_dim)`.
    The pooled outputs are mean-pooled over each section's token range; sections
    that were split into fragments are recombined by token-count-weighted average
    across their fragments.
    """
    fragment_sums: dict[tuple[int, int], np.ndarray] = {}
    fragment_counts: dict[tuple[int, int], int] = {}
    for window, outputs in zip(windows, window_token_outputs, strict=True):
        if outputs.shape != (window.length, embedding_dim):
            raise ValueError(
                f"window output shape {outputs.shape} != ({window.length}, {embedding_dim})"
            )
        for segment in window.core_segments:
            window_start, window_end = window.core_segment_window_range(segment)
            segment_outputs = outputs[window_start:window_end]
            if segment_outputs.shape[0] == 0:
                continue
            key = (segment.section_index, segment.fragment_index)
            fragment_sums[key] = segment_outputs.sum(axis=0).astype(np.float32)
            fragment_counts[key] = segment_outputs.shape[0]

    section_vectors = np.zeros((n_sections, embedding_dim), dtype=np.float32)
    section_token_totals = np.zeros(n_sections, dtype=np.int64)
    for (section_index, _fragment_index), summed in fragment_sums.items():
        n_tokens = fragment_counts[(section_index, _fragment_index)]
        section_vectors[section_index] += summed
        section_token_totals[section_index] += n_tokens
    nonzero = section_token_totals > 0
    section_vectors[nonzero] /= section_token_totals[nonzero, None]
    return section_vectors


def chunk_article(
    text: str,
    tokenizer,  # transformers.PreTrainedTokenizer
    context_limit: int = 8192,
    margin: int = 256,
) -> tuple[list[Window], list[SectionTokenSpan]]:
    """End-to-end: rendered Markdown -> windows ready for the model.

    Returns the windows plus the section token spans (so callers can correlate
    section vectors back to heading metadata)."""
    char_spans = find_section_char_spans(text)
    encoding = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    token_ids: list[int] = encoding["input_ids"]
    token_offsets: list[tuple[int, int]] = list(encoding["offset_mapping"])
    section_spans = section_token_spans_from_offsets(char_spans, token_offsets)
    windows = plan_windows(token_ids, section_spans, context_limit, margin)
    return windows, section_spans
