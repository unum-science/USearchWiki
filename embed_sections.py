"""Late-chunking section embeddings via GTE-ModernColBERT-v1 (pylate).

For each parquet shard:
  1. Load articles (text + id).
  2. For each article: find section char spans, tokenize ("[D] " + text)
     once, plan windows via late_chunking, forward each window through the
     transformer + projection + L2 normalization, mean-pool the core token
     vectors per section.
  3. Write per-shard `{wiki}/{stem}.body.sections.f16bin` (concatenated section
     embeddings) and `{wiki}/{stem}.body.sections.offsets.ibin` (cumulative
     offsets giving each article's section slice).

One process per GPU; partition shards by `shard_index % world_size == gpu_id`.

Usage:
  CUDA_VISIBLE_DEVICES=0 python embed_sections.py \\
      --cache-dir /home/ubuntu/wikiverse-data/hf-cache \\
      --output /home/ubuntu/WikiVerse \\
      --model-subdir gte-moderncolbert-v1 \\
      --wiki enwiki --gpu-id 0 --world-size 8
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from late_chunking import (  # noqa: E402
    SectionCharSpan,
    Window,
    find_section_char_spans,
    plan_windows,
    pool_section_vectors,
    section_token_spans_from_offsets,
)
from wikiverse import Shard, load_lang  # noqa: E402


def select_shards(all_shards: list[Shard], gpu_id: int, world_size: int) -> list[Shard]:
    return [s for i, s in enumerate(all_shards) if i % world_size == gpu_id]


def load_pylate_model(model_id: str, device: str, document_length: int):
    from pylate import models  # local import (heavy)

    print(f"[gpu] loading {model_id} on {device} ...", flush=True)
    started = time.monotonic()
    model = models.ColBERT(
        model_name_or_path=model_id,
        device=device,
        document_length=document_length,
    )
    # Pylate defaults to fp32; switching to fp16 is ~3x faster with no
    # measurable quality loss for this model. Cast both submodules.
    model[0].auto_model = model[0].auto_model.half()
    model[1].linear = model[1].linear.half()
    model.eval()
    print(
        f"[gpu] loaded in {time.monotonic() - started:.1f}s "
        f"(fp16: transformer={next(model[0].auto_model.parameters()).dtype}, "
        f"dense={next(model[1].linear.parameters()).dtype})",
        flush=True,
    )
    return model


def plan_articles_batched(
    texts: list[str],
    tokenizer,
    document_prefix: str,
    context_limit: int,
    margin: int,
) -> tuple[list[list[Window]], list[int]]:
    """Tokenize all articles in one tokenizer call (fast Rust path), then plan
    windows per article. Returns (per_article_windows, per_article_n_sections).
    """
    n = len(texts)
    per_article_char_spans: list[list[SectionCharSpan]] = []
    prefixed_texts: list[str] = []
    nonempty_indices: list[int] = []
    for index, text in enumerate(texts):
        if not text:
            per_article_char_spans.append([])
            continue
        char_spans = find_section_char_spans(text)
        if not char_spans:
            per_article_char_spans.append([])
            continue
        per_article_char_spans.append(char_spans)
        prefixed_texts.append(document_prefix + text)
        nonempty_indices.append(index)

    per_article_windows: list[list[Window]] = [[] for _ in range(n)]
    per_article_n_sections: list[int] = [0] * n

    if not prefixed_texts:
        return per_article_windows, per_article_n_sections

    # One Rust-side tokenizer call for the whole batch.
    encodings = tokenizer(
        prefixed_texts,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )

    prefix_len = len(document_prefix)
    for batch_index, article_index in enumerate(nonempty_indices):
        token_ids = encodings["input_ids"][batch_index]
        offsets = encodings["offset_mapping"][batch_index]
        char_spans = per_article_char_spans[article_index]
        shifted = [
            SectionCharSpan(
                char_start=s.char_start + prefix_len,
                char_end=s.char_end + prefix_len,
                heading_level=s.heading_level,
                heading_text=s.heading_text,
            )
            for s in char_spans
        ]
        section_spans = section_token_spans_from_offsets(shifted, list(offsets))
        if not section_spans:
            continue
        windows = plan_windows(token_ids, section_spans, context_limit, margin)
        per_article_windows[article_index] = windows
        per_article_n_sections[article_index] = len(section_spans)

    return per_article_windows, per_article_n_sections


def encode_articles_batch(
    texts: list[str],
    model,
    cls_id: int,
    sep_id: int,
    pad_id: int,
    device: str,
    context_limit: int,
    margin: int,
    document_prefix: str,
    max_batch_tokens: int,
) -> list[np.ndarray]:
    """Encode a batch of articles into per-article (n_sections, dim) FP16 arrays.

    Pads windows from across the batch into one or more padded forward passes,
    splitting into sub-batches whenever the padded total exceeds
    `max_batch_tokens` (so a few very long articles don't blow up GPU memory).
    """
    embedding_dim = model[1].linear.out_features

    # Plan windows for every article via a single batched tokenizer call.
    per_article_windows, per_article_n_sections = plan_articles_batched(
        texts=texts,
        tokenizer=model.tokenizer,
        document_prefix=document_prefix,
        context_limit=context_limit,
        margin=margin,
    )

    # Flatten window list across articles, tag each with its article index.
    all_windows: list[tuple[int, int, Window]] = []
    for article_index, windows in enumerate(per_article_windows):
        for window_index, window in enumerate(windows):
            all_windows.append((article_index, window_index, window))

    # Outputs scratch: one numpy array per (article, window) once we have it.
    output_token_arrays: dict[tuple[int, int], np.ndarray] = {}

    if all_windows:
        # Sort windows by length to keep padding overhead per sub-batch low.
        all_windows.sort(key=lambda triple: triple[2].length)

        sub_batch: list[tuple[int, int, Window]] = []
        sub_batch_max_len = 0

        def flush(sub: list[tuple[int, int, Window]]) -> None:
            if not sub:
                return
            wrapped_max = max(triple[2].length for triple in sub) + 2
            input_ids = torch.full(
                (len(sub), wrapped_max), pad_id, dtype=torch.long, device=device
            )
            attention_mask = torch.zeros(
                (len(sub), wrapped_max), dtype=torch.long, device=device
            )
            for row, (_, _, window) in enumerate(sub):
                wrapped = [cls_id] + window.token_ids + [sep_id]
                input_ids[row, : len(wrapped)] = torch.tensor(
                    wrapped, dtype=torch.long, device=device
                )
                attention_mask[row, : len(wrapped)] = 1
            with torch.inference_mode():
                hidden = (
                    model[0]
                    .auto_model(input_ids=input_ids, attention_mask=attention_mask)
                    .last_hidden_state
                )
                projected = model[1].linear(hidden)
                normalized = F.normalize(projected, p=2, dim=-1)
            for row, (article_index, window_index, window) in enumerate(sub):
                out = (
                    normalized[row, 1 : 1 + window.length, :]
                    .to(torch.float32)
                    .cpu()
                    .numpy()
                )
                output_token_arrays[(article_index, window_index)] = out

        for triple in all_windows:
            window = triple[2]
            wrapped_len = window.length + 2
            new_max_len = max(sub_batch_max_len, wrapped_len)
            projected_padded_tokens = (len(sub_batch) + 1) * new_max_len
            if sub_batch and projected_padded_tokens > max_batch_tokens:
                flush(sub_batch)
                sub_batch = []
                sub_batch_max_len = 0
            sub_batch.append(triple)
            sub_batch_max_len = max(sub_batch_max_len, wrapped_len)
        flush(sub_batch)

    # Pool per article.
    section_matrices: list[np.ndarray] = []
    for article_index, (windows, n_sections) in enumerate(
        zip(per_article_windows, per_article_n_sections, strict=True)
    ):
        if n_sections == 0:
            section_matrices.append(np.zeros((0, embedding_dim), dtype=np.float16))
            continue
        token_outputs = [
            output_token_arrays[(article_index, window_index)]
            for window_index in range(len(windows))
        ]
        section_matrix = pool_section_vectors(
            windows=windows,
            window_token_outputs=token_outputs,
            n_sections=n_sections,
            embedding_dim=embedding_dim,
        )
        norms = np.linalg.norm(section_matrix, axis=1, keepdims=True)
        nonzero = norms[:, 0] > 0
        section_matrix[nonzero] = section_matrix[nonzero] / norms[nonzero]
        section_matrices.append(section_matrix.astype(np.float16))
    return section_matrices


def write_shard_outputs(
    shard_dir: Path,
    stem: str,
    suffix: str,
    section_matrices: list[np.ndarray],
    embedding_dim: int,
) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    section_counts = [m.shape[0] for m in section_matrices]
    total_sections = sum(section_counts)
    cumulative_offsets = np.zeros(len(section_matrices) + 1, dtype=np.int32)
    cumulative_offsets[1:] = np.cumsum(section_counts, dtype=np.int32)

    sections_path = shard_dir / f"{stem}.{suffix}.sections.f16bin"
    offsets_path = shard_dir / f"{stem}.{suffix}.sections.offsets.ibin"

    with open(sections_path.with_suffix(sections_path.suffix + ".tmp"), "wb") as file:
        file.write(struct.pack("<II", total_sections, embedding_dim))
        if total_sections > 0:
            concatenated = (
                np.vstack(section_matrices)
                if section_matrices
                else np.zeros((0, embedding_dim), dtype=np.float16)
            )
            file.write(concatenated.tobytes(order="C"))
    sections_path.with_suffix(sections_path.suffix + ".tmp").rename(sections_path)

    with open(offsets_path.with_suffix(offsets_path.suffix + ".tmp"), "wb") as file:
        file.write(struct.pack("<II", len(cumulative_offsets), 1))
        file.write(cumulative_offsets.tobytes(order="C"))
    offsets_path.with_suffix(offsets_path.suffix + ".tmp").rename(offsets_path)


def process_shard(
    shard: Shard,
    output_root: Path,
    model,
    cls_id: int,
    sep_id: int,
    pad_id: int,
    device: str,
    context_limit: int,
    margin: int,
    document_prefix: str,
    suffix: str,
    text_column: str,
    id_column: str,
    article_batch_size: int,
    max_batch_tokens: int,
) -> dict:
    table = pq.read_table(shard.path, columns=[id_column, text_column])
    ids = table.column(id_column).to_pylist()
    texts = table.column(text_column).to_pylist()

    section_matrices: list[np.ndarray] = []
    n_sections_total = 0
    n_zero = 0
    started = time.monotonic()
    progress_every = max(article_batch_size, len(ids) // 20)
    embedding_dim = model[1].linear.out_features

    for batch_start in range(0, len(ids), article_batch_size):
        batch_end = min(batch_start + article_batch_size, len(ids))
        batch_texts = [t or "" for t in texts[batch_start:batch_end]]
        try:
            batch_matrices = encode_articles_batch(
                texts=batch_texts,
                model=model,
                cls_id=cls_id,
                sep_id=sep_id,
                pad_id=pad_id,
                device=device,
                context_limit=context_limit,
                margin=margin,
                document_prefix=document_prefix,
                max_batch_tokens=max_batch_tokens,
            )
        except Exception as exc:
            print(
                f"  ! batch [{batch_start},{batch_end}) failed: {exc!r}; "
                f"emitting zero sections for the batch",
                flush=True,
            )
            batch_matrices = [
                np.zeros((0, embedding_dim), dtype=np.float16) for _ in batch_texts
            ]
        for matrix in batch_matrices:
            section_matrices.append(matrix)
            n_sections_total += matrix.shape[0]
            if matrix.shape[0] == 0:
                n_zero += 1
        if batch_end % progress_every < article_batch_size or batch_end == len(ids):
            elapsed = time.monotonic() - started
            rate = batch_end / max(elapsed, 1e-3)
            print(
                f"    {shard.wikiname}/{shard.stem}: {batch_end}/{len(ids)} articles "
                f"({rate:.1f} doc/s, {n_sections_total:,} sections so far)",
                flush=True,
            )

    elapsed = time.monotonic() - started
    embedding_dim = model[1].linear.out_features

    write_shard_outputs(
        shard_dir=output_root / shard.wikiname,
        stem=shard.stem,
        suffix=suffix,
        section_matrices=section_matrices,
        embedding_dim=embedding_dim,
    )

    return {
        "n_articles": len(ids),
        "n_zero_articles": n_zero,
        "n_sections_total": n_sections_total,
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="/home/ubuntu/wikiverse-data/hf-cache")
    parser.add_argument("--output", default="/home/ubuntu/WikiVerse")
    parser.add_argument("--model-subdir", default="gte-moderncolbert-v1")
    parser.add_argument("--model-id", default="lightonai/GTE-ModernColBERT-v1")
    parser.add_argument(
        "--wiki", required=True, help="single language code (enwiki, dewiki, ...)"
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--context-limit", type=int, default=8192)
    parser.add_argument("--margin", type=int, default=256)
    parser.add_argument("--text-column", default="text", choices=["text", "title"])
    parser.add_argument("--output-suffix", default="body")
    parser.add_argument("--id-column", default="id")
    parser.add_argument(
        "--article-batch-size",
        type=int,
        default=64,
        help="number of articles' windows to plan in one cross-article batch",
    )
    parser.add_argument(
        "--max-batch-tokens",
        type=int,
        default=131072,
        help="cap on `padded_batch_size * padded_max_length` per forward pass; "
        "splits the cross-article batch into sub-batches when needed to keep "
        "GPU memory bounded",
    )
    args = parser.parse_args()

    # Pin to the assigned GPU before importing pylate.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu_id))
    device = "cuda:0"  # post-CUDA_VISIBLE_DEVICES, our GPU is always cuda:0.

    output_root = Path(args.output) / args.model_subdir
    output_root.mkdir(parents=True, exist_ok=True)

    shards = load_lang(args.cache_dir, args.wiki)
    owned = select_shards(shards, args.gpu_id, args.world_size)
    pending = [
        s
        for s in owned
        if not (
            output_root / s.wikiname / f"{s.stem}.{args.output_suffix}.sections.f16bin"
        ).exists()
    ]
    print(
        f"[gpu{args.gpu_id}] {args.wiki}: {len(owned)} owned, {len(pending)} pending",
        flush=True,
    )
    if not pending:
        return

    model = load_pylate_model(args.model_id, device, args.context_limit)
    cls_id = model.tokenizer.cls_token_id
    sep_id = model.tokenizer.sep_token_id
    pad_id = model.tokenizer.pad_token_id
    document_prefix = model.document_prefix

    started_overall = time.monotonic()
    total_articles = 0
    total_sections = 0
    for shard in pending:
        stats = process_shard(
            shard=shard,
            output_root=output_root,
            model=model,
            cls_id=cls_id,
            sep_id=sep_id,
            pad_id=pad_id,
            device=device,
            context_limit=args.context_limit,
            margin=args.margin,
            document_prefix=document_prefix,
            suffix=args.output_suffix,
            text_column=args.text_column,
            id_column=args.id_column,
            article_batch_size=args.article_batch_size,
            max_batch_tokens=args.max_batch_tokens,
        )
        total_articles += stats["n_articles"]
        total_sections += stats["n_sections_total"]
        rate = stats["n_articles"] / max(stats["elapsed_seconds"], 1e-3)
        print(
            f"[gpu{args.gpu_id} pylate] {shard.wikiname}/{shard.stem}: "
            f"{stats['n_articles']} articles ({stats['n_zero_articles']} zero), "
            f"{stats['n_sections_total']:,} sections in {stats['elapsed_seconds']:.1f}s "
            f"-> {rate:.1f} doc/s",
            flush=True,
        )

    wall = time.monotonic() - started_overall
    print(
        f"[gpu{args.gpu_id} pylate] DONE: {total_articles} articles, "
        f"{total_sections:,} sections in {wall:.0f}s -> {total_articles/max(wall,1):.1f} doc/s",
        flush=True,
    )


if __name__ == "__main__":
    main()
