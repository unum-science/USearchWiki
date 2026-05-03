"""Late-chunking section embeddings via GTE-ModernColBERT-v1 (pylate).

Pool scheduler: enumerates every (wiki, shard) tuple in the corpus, filters
out already-completed outputs, and dispatches the rest across `num_gpus`
long-running worker processes via a shared `mp.Queue`. Each worker loads the
pylate model once and processes shards as they arrive, so small single-shard
wikis don't leave 7 GPUs idle.

For each parquet shard a worker:
  1. Loads articles (text + id).
  2. Per article: finds section char spans, tokenizes ("[D] " + text) once,
     plans windows via `late_chunking`, forwards each window through the
     transformer + projection + L2 normalization, mean-pools the core token
     vectors per section.
  3. Writes per-shard `{wiki}/{stem}.body.sections.f16bin` (concatenated
     section embeddings) and `{wiki}/{stem}.body.sections.offsets.ibin`
     (cumulative offsets giving each article's section slice).

Resume-safe by construction: shards whose output files already exist are
skipped before being added to the queue.

Usage:
  python embed_sections.py \\
      --cache-dir /home/ubuntu/wikiverse-data/hf-cache \\
      --output /home/ubuntu/USearchWiki \\
      --model-subdir gte-moderncolbert-v1 \\
      --num-gpus 8
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
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
from usearchwiki import Shard, find_snapshot, load_lang  # noqa: E402


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
    """Encode a batch of articles into per-article (n_sections, dim) FP16 arrays."""
    embedding_dim = model[1].linear.out_features

    per_article_windows, per_article_n_sections = plan_articles_batched(
        texts=texts,
        tokenizer=model.tokenizer,
        document_prefix=document_prefix,
        context_limit=context_limit,
        margin=margin,
    )

    all_windows: list[tuple[int, int, Window]] = []
    for article_index, windows in enumerate(per_article_windows):
        for window_index, window in enumerate(windows):
            all_windows.append((article_index, window_index, window))

    output_token_arrays: dict[tuple[int, int], np.ndarray] = {}

    if all_windows:
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


def worker(gpu_id: int, work_queue, args_dict: dict) -> None:
    """Pin to one GPU, load the model once, drain shards from the queue."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"
    model = load_pylate_model(
        args_dict["model_id"], device, args_dict["context_limit"]
    )
    cls_id = model.tokenizer.cls_token_id
    sep_id = model.tokenizer.sep_token_id
    pad_id = model.tokenizer.pad_token_id
    document_prefix = model.document_prefix
    output_root = Path(args_dict["output"]) / args_dict["model_subdir"]

    n_processed = 0
    n_failed = 0
    started = time.monotonic()
    while True:
        try:
            item = work_queue.get(timeout=5.0)
        except Exception:
            print(f"[gpu{gpu_id}] queue idle 5s, exiting", flush=True)
            break
        if item is None:
            break
        shard: Shard = item
        try:
            stats = process_shard(
                shard=shard,
                output_root=output_root,
                model=model,
                cls_id=cls_id,
                sep_id=sep_id,
                pad_id=pad_id,
                device=device,
                context_limit=args_dict["context_limit"],
                margin=args_dict["margin"],
                document_prefix=document_prefix,
                suffix=args_dict["suffix"],
                text_column=args_dict["text_column"],
                id_column=args_dict["id_column"],
                article_batch_size=args_dict["article_batch_size"],
                max_batch_tokens=args_dict["max_batch_tokens"],
            )
            n_processed += 1
            rate = stats["n_articles"] / max(stats["elapsed_seconds"], 1e-3)
            print(
                f"[gpu{gpu_id}] {shard.wikiname}/{shard.stem}: "
                f"{stats['n_articles']} articles "
                f"({stats['n_zero_articles']} zero), "
                f"{stats['n_sections_total']:,} sections in "
                f"{stats['elapsed_seconds']:.0f}s -> {rate:.1f} doc/s",
                flush=True,
            )
        except Exception as exc:
            n_failed += 1
            print(
                f"[gpu{gpu_id}] {shard.wikiname}/{shard.stem}: FAILED: {exc!r}",
                flush=True,
            )

    elapsed = time.monotonic() - started
    print(
        f"[gpu{gpu_id}] worker DONE: {n_processed} shards processed, "
        f"{n_failed} failed, {elapsed:.0f}s",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="/home/ubuntu/wikiverse-data/hf-cache")
    parser.add_argument("--output", default="/home/ubuntu/USearchWiki")
    parser.add_argument("--model-subdir", default="gte-moderncolbert-v1")
    parser.add_argument("--model-id", default="lightonai/GTE-ModernColBERT-v1")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--context-limit", type=int, default=8192)
    parser.add_argument("--margin", type=int, default=256)
    parser.add_argument("--text-column", default="text", choices=["text", "title"])
    parser.add_argument("--output-suffix", default="body")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--article-batch-size", type=int, default=64)
    parser.add_argument("--max-batch-tokens", type=int, default=131072)
    args = parser.parse_args()

    snapshot = find_snapshot(args.cache_dir)
    wiki_names = sorted(
        d.name for d in (snapshot / "data").iterdir() if d.is_dir()
    )
    print(
        f"discovering shards across {len(wiki_names)} wikis under "
        f"{snapshot.parent.name}/{snapshot.name} ...",
        flush=True,
    )

    output_root = Path(args.output) / args.model_subdir
    pending: list[Shard] = []
    skipped = 0
    for wiki_name in wiki_names:
        try:
            shards = load_lang(args.cache_dir, wiki_name)
        except Exception:
            continue
        for shard in shards:
            existing = (
                output_root
                / shard.wikiname
                / f"{shard.stem}.{args.output_suffix}.sections.f16bin"
            )
            if existing.exists():
                skipped += 1
                continue
            pending.append(shard)
    pending.sort(key=lambda shard: shard.path.stat().st_size, reverse=True)
    print(
        f"  {skipped} shards already done; {len(pending)} pending; "
        f"largest parquet: {pending[0].path.stat().st_size / 1e6:.0f} MB"
        if pending
        else f"  {skipped} shards already done; nothing pending",
        flush=True,
    )
    if not pending:
        return

    args_dict = {
        "model_id": args.model_id,
        "context_limit": args.context_limit,
        "margin": args.margin,
        "output": args.output,
        "model_subdir": args.model_subdir,
        "suffix": args.output_suffix,
        "text_column": args.text_column,
        "id_column": args.id_column,
        "article_batch_size": args.article_batch_size,
        "max_batch_tokens": args.max_batch_tokens,
    }

    ctx = mp.get_context("fork")
    work_queue: mp.Queue = ctx.Queue()
    for shard in pending:
        work_queue.put(shard)
    for _ in range(args.num_gpus):
        work_queue.put(None)

    workers: list[mp.Process] = []
    for gpu_id in range(args.num_gpus):
        process = ctx.Process(target=worker, args=(gpu_id, work_queue, args_dict))
        process.start()
        workers.append(process)

    started = time.monotonic()
    print(
        f"started {len(workers)} GPU workers at "
        f"{time.strftime('%Y-%m-%dT%H:%M:%S')}; "
        f"{len(pending)} shards in queue",
        flush=True,
    )

    failed = 0
    for process in workers:
        process.join()
        if process.exitcode != 0:
            failed += 1
            print(
                f"  worker pid {process.pid} exited code {process.exitcode}",
                flush=True,
            )
    elapsed = time.monotonic() - started
    print(
        f"DONE: {len(pending)} shards in {elapsed:.0f}s "
        f"({len(pending) / max(elapsed, 1e-3):.2f} shards/s); "
        f"{failed} workers failed",
        flush=True,
    )


if __name__ == "__main__":
    main()
