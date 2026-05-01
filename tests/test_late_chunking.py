"""Synthetic tests for late_chunking.py.

The chunker is purely arithmetic — given an article's tokens and section
spans, it produces windows. We can verify all the invariants without a model:

  - every section's core tokens appear in exactly one window
    (or, when split, every section's core tokens are partitioned across
    fragments without gaps or overlaps)
  - every window's input length is at most context_limit
  - every section's pooled token range falls inside a window's core
  - margins respect article boundaries
  - pool_section_vectors recovers the right per-section means
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from late_chunking import (  # noqa: E402
    CoreSegment,
    SectionCharSpan,
    SectionTokenSpan,
    Window,
    find_section_char_spans,
    plan_windows,
    pool_section_vectors,
    section_token_spans_from_offsets,
)


def make_synthetic_section_spans(
    section_sizes: list[int],
) -> tuple[list[int], list[SectionTokenSpan]]:
    """Build (article_token_ids, section_token_spans) where each token's id
    encodes its (section_index, position_in_section) so we can verify that
    pooling maps the right tokens back to the right section."""
    spans: list[SectionTokenSpan] = []
    token_ids: list[int] = []
    cursor = 0
    for index, size in enumerate(section_sizes):
        spans.append(
            SectionTokenSpan(
                section_index=index,
                token_start=cursor,
                token_end=cursor + size,
                heading_level=2,
                heading_text=f"Section {index}",
            )
        )
        for position in range(size):
            # Encode (section_index, position) so the test can verify pooling.
            token_ids.append(index * 100000 + position)
        cursor += size
    return token_ids, spans


def assert_windows_cover_all_sections(
    section_sizes: list[int],
    spans: list[SectionTokenSpan],
    windows: list[Window],
) -> None:
    """Every section's [token_start, token_end) must be covered by exactly one
    set of CoreSegments whose article ranges concatenate to the original."""
    by_section: dict[int, list[CoreSegment]] = {}
    for window in windows:
        for segment in window.core_segments:
            by_section.setdefault(segment.section_index, []).append(segment)
    for index, span in enumerate(spans):
        section_segments = sorted(
            by_section[index], key=lambda seg: seg.fragment_index
        )
        # Fragments must form a contiguous partition.
        cursor = span.token_start
        for fragment in section_segments:
            assert fragment.article_token_start == cursor, (
                f"section {index} fragment gap at {cursor} vs {fragment.article_token_start}"
            )
            cursor = fragment.article_token_end
            assert fragment.fragment_count == len(section_segments), (
                f"section {index} fragment_count mismatch"
            )
        assert cursor == span.token_end, (
            f"section {index} fragments end at {cursor}, expected {span.token_end}"
        )


def assert_window_lengths_within_limit(
    windows: list[Window], context_limit: int
) -> None:
    for index, window in enumerate(windows):
        assert window.length <= context_limit, (
            f"window {index} length {window.length} exceeds context_limit {context_limit}"
        )
        # Core boundaries must lie within the window.
        assert 0 <= window.core_start <= window.core_end <= window.length


def assert_segment_offsets_consistent(windows: list[Window]) -> None:
    for window in windows:
        for segment in window.core_segments:
            window_start, window_end = window.core_segment_window_range(segment)
            assert window.core_start <= window_start < window_end <= window.core_end, (
                f"segment {segment} offsets ({window_start}, {window_end}) "
                f"escape core [{window.core_start}, {window.core_end})"
            )


def test_single_section_fits_in_one_window() -> None:
    section_sizes = [400]
    article_tokens, spans = make_synthetic_section_spans(section_sizes)
    windows = plan_windows(article_tokens, spans, context_limit=8192, margin=256)
    assert len(windows) == 1
    assert_window_lengths_within_limit(windows, 8192)
    assert_windows_cover_all_sections(section_sizes, spans, windows)
    # No left margin (article starts here) and no right margin (article ends here).
    assert windows[0].core_start == 0
    assert windows[0].core_end == 400
    assert windows[0].length == 400
    print("PASS: single-section fits in one window")


def test_many_small_sections_pack_into_one_window() -> None:
    section_sizes = [100] * 30  # 3000 total tokens
    article_tokens, spans = make_synthetic_section_spans(section_sizes)
    windows = plan_windows(article_tokens, spans, context_limit=8192, margin=256)
    assert len(windows) == 1, f"expected 1 window, got {len(windows)}"
    assert sum(s.token_end - s.token_start for s in spans) == windows[0].core_end - windows[0].core_start
    print(f"PASS: 30 small sections packed into 1 window ({windows[0].length} tokens)")


def test_sections_split_across_multiple_windows() -> None:
    # 10 sections of 1000 each = 10K tokens; with core 7680 that's 2 windows.
    section_sizes = [1000] * 10
    article_tokens, spans = make_synthetic_section_spans(section_sizes)
    windows = plan_windows(article_tokens, spans, context_limit=8192, margin=256)
    assert len(windows) == 2, f"expected 2 windows, got {len(windows)}"
    assert_window_lengths_within_limit(windows, 8192)
    assert_windows_cover_all_sections(section_sizes, spans, windows)
    assert_segment_offsets_consistent(windows)
    # Window 0 has no left margin (article start); has a right margin from window 1's first section.
    assert windows[0].article_left_margin_start == 0
    assert windows[0].length > windows[0].core_end - windows[0].core_start  # right margin present
    # Window 1 has a left margin from window 0's tail; no right margin (article end).
    assert windows[1].article_right_margin_end == 10000
    assert windows[1].core_start > 0  # left margin present
    print(f"PASS: 10 x 1000-token sections -> {len(windows)} windows with margins")


def test_oversized_section_split_into_fragments() -> None:
    # One section of 20K tokens, where context=8192, margin=256, core_max=7680.
    # 20000 / 7680 = 3 fragments (sizes 6667, 6667, 6666).
    section_sizes = [20000]
    article_tokens, spans = make_synthetic_section_spans(section_sizes)
    windows = plan_windows(article_tokens, spans, context_limit=8192, margin=256)
    assert len(windows) == 3, f"expected 3 windows, got {len(windows)}"
    fragments = [seg for w in windows for seg in w.core_segments]
    assert len(fragments) == 3
    assert all(f.fragment_count == 3 for f in fragments)
    # Fragments must partition the section's token range.
    starts = [f.article_token_start for f in fragments]
    ends = [f.article_token_end for f in fragments]
    assert starts[0] == 0 and ends[-1] == 20000
    for i in range(len(fragments) - 1):
        assert ends[i] == starts[i + 1]
    assert_window_lengths_within_limit(windows, 8192)
    print("PASS: oversized section split into 3 fragments across 3 windows")


def test_mixed_normal_and_oversized_sections() -> None:
    # Lead 100, big middle 10000, tail 200.
    section_sizes = [100, 10000, 200]
    article_tokens, spans = make_synthetic_section_spans(section_sizes)
    windows = plan_windows(article_tokens, spans, context_limit=8192, margin=256)
    assert_windows_cover_all_sections(section_sizes, spans, windows)
    assert_window_lengths_within_limit(windows, 8192)
    assert_segment_offsets_consistent(windows)
    # The middle section needs 2 windows to itself; the lead and tail go in
    # their own windows because we don't share with split sections.
    fragment_counts = {seg.section_index: seg.fragment_count for w in windows for seg in w.core_segments}
    assert fragment_counts == {0: 1, 1: 2, 2: 1}, fragment_counts
    print(f"PASS: mixed normal+oversized -> {len(windows)} windows")


def test_pool_section_vectors_recovers_means() -> None:
    """Synthesize per-token outputs whose mean is the section index;
    verify that pooled vectors equal the section index."""
    section_sizes = [50, 30, 20]
    article_tokens, spans = make_synthetic_section_spans(section_sizes)
    windows = plan_windows(article_tokens, spans, context_limit=8192, margin=256)
    embedding_dim = 8
    window_outputs: list[np.ndarray] = []
    for window in windows:
        out = np.zeros((window.length, embedding_dim), dtype=np.float32)
        for segment in window.core_segments:
            wstart, wend = window.core_segment_window_range(segment)
            # Fill core token rows with a vector whose values equal section_index.
            out[wstart:wend] = float(segment.section_index) + 0.0
        window_outputs.append(out)
    pooled = pool_section_vectors(windows, window_outputs, n_sections=3, embedding_dim=embedding_dim)
    expected = np.array(
        [[0.0] * 8, [1.0] * 8, [2.0] * 8],
        dtype=np.float32,
    )
    assert np.allclose(pooled, expected), pooled
    print("PASS: pool_section_vectors recovers per-section means")


def test_pool_recovers_split_section_via_weighted_mean() -> None:
    """Section split into 2 fragments of different sizes; per-fragment mean is
    different; pooled section vector must be the token-weighted average."""
    # Single section, 12K tokens — splits into 2 fragments of 6K each (since
    # core max = 7680 and 12000/7680 rounds to 2).
    section_sizes = [12000]
    article_tokens, spans = make_synthetic_section_spans(section_sizes)
    windows = plan_windows(article_tokens, spans, context_limit=8192, margin=256)
    assert len(windows) == 2
    embedding_dim = 4
    window_outputs: list[np.ndarray] = []
    for window in windows:
        out = np.zeros((window.length, embedding_dim), dtype=np.float32)
        for segment in window.core_segments:
            wstart, wend = window.core_segment_window_range(segment)
            # Fragment 0 -> mean is 1.0; fragment 1 -> mean is 5.0.
            value = 1.0 if segment.fragment_index == 0 else 5.0
            out[wstart:wend] = value
        window_outputs.append(out)
    pooled = pool_section_vectors(windows, window_outputs, n_sections=1, embedding_dim=embedding_dim)
    # Both fragments are 6000 tokens; weighted mean = (1*6000 + 5*6000)/12000 = 3.0.
    assert np.allclose(pooled, np.full((1, 4), 3.0)), pooled
    print("PASS: split section recovered via token-weighted average")


def test_section_char_spans_lead_only() -> None:
    text = "Just one paragraph of text, no headings."
    spans = find_section_char_spans(text)
    assert len(spans) == 1
    assert spans[0].heading_text is None
    assert spans[0].heading_level == 0
    print("PASS: lead-only article -> 1 section span")


def test_section_char_spans_with_headings() -> None:
    text = "Lead paragraph.\n\n# History\nHistory text.\n\n## Origins\nOrigins text.\n"
    spans = find_section_char_spans(text)
    levels_and_titles = [(s.heading_level, s.heading_text) for s in spans]
    assert levels_and_titles == [(0, None), (1, "History"), (2, "Origins")], levels_and_titles
    # The last section ends at end-of-text.
    assert spans[-1].char_end == len(text)
    print("PASS: lead + 2 headings -> 3 section spans")


def test_section_token_spans_from_offsets() -> None:
    # Two sections at char [0,100) and [100, 200); offsets every 5 chars.
    char_spans = [
        SectionCharSpan(char_start=0, char_end=100, heading_level=0, heading_text=None),
        SectionCharSpan(char_start=100, char_end=200, heading_level=1, heading_text="X"),
    ]
    offsets = [(i * 5, i * 5 + 5) for i in range(40)]
    token_spans = section_token_spans_from_offsets(char_spans, offsets)
    assert len(token_spans) == 2
    assert token_spans[0].token_start == 0 and token_spans[0].token_end == 20
    assert token_spans[1].token_start == 20 and token_spans[1].token_end == 40
    print("PASS: char->token span conversion")


if __name__ == "__main__":
    test_section_char_spans_lead_only()
    test_section_char_spans_with_headings()
    test_section_token_spans_from_offsets()
    test_single_section_fits_in_one_window()
    test_many_small_sections_pack_into_one_window()
    test_sections_split_across_multiple_windows()
    test_oversized_section_split_into_fragments()
    test_mixed_normal_and_oversized_sections()
    test_pool_section_vectors_recovers_means()
    test_pool_recovers_split_section_via_weighted_mean()
    print("\nALL TESTS PASSED")
