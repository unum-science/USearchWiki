"""Compute exact global k-NN ground truth for an embedding collection.

Two modes share the same per-stripe partition + per-shard output pipeline:

  --mode dense  (default)
    Cosine top-k over a `(N, dim)` FP16 corpus. Used for the
    article-level dense models (qwen3-embedding-0.6b,
    snowflake-arctic-embed-l-v2.0, nomic-embed-text-v1.5, ...).

  --mode maxsim
    ColBERT-style late-interaction MaxSim top-k over a
    `(T_total, dim)` token bank + `(N+1,)` section offsets. Used for
    the section-level multi-vector models (gte-moderncolbert-v1).

Both modes write per-shard `{wiki}/{stem}.{suffix}.ground_truth.{ibin,fbin}`
files in canonical shard-walk order. The neighbor IDs in the `.ibin` are
global row IDs (article-id space for dense, section-id space for MaxSim).

Usage:
  python ground_truth.py --mode dense \
      --output /path/to/embeddings --model-subdir qwen3-embedding-0.6b \
      --num-gpus 8

  python ground_truth.py --mode maxsim \
      --output /path/to/embeddings --model-subdir gte-moderncolbert-v1 \
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

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from retrievers import (  # noqa: E402
    load_maxsim_corpus,
    gt_stripe_dense,
    gt_stripe_maxsim,
)
from usearchwiki import (  # noqa: E402
    CollectionShard,
    discover_collection,
    resolve_lfs_pointer,
    write_bin,
)


def load_dense_collection(
    model_root: Path,
    suffix: str,
    dimensions: int,
    shards: list[CollectionShard],
) -> np.ndarray:
    total_vectors = sum(shard.row_count for shard in shards)
    embeddings = np.empty((total_vectors, dimensions), dtype=np.float16)
    started = time.monotonic()
    for shard in shards:
        path = model_root / shard.wikiname / f"{shard.stem}.{suffix}.f16bin"
        blob_path = resolve_lfs_pointer(path)
        with open(blob_path, "rb") as file:
            header = file.read(8)
            rows, columns = struct.unpack("<II", header)
            if columns != dimensions:
                raise ValueError(f"{path}: dim {columns} != expected {dimensions}")
            if rows != shard.row_count:
                raise ValueError(f"{path}: rows {rows} != cached {shard.row_count}")
            destination = embeddings[shard.row_offset : shard.row_offset + rows]
            file.readinto(memoryview(destination))  # type: ignore[arg-type]
            del destination
    elapsed = time.monotonic() - started
    gigabytes = embeddings.nbytes / 1e9
    print(
        f"loaded {total_vectors:,} x {dimensions} fp16 ({gigabytes:.1f} GB) "
        f"from {len(shards)} shards in {elapsed:.1f}s",
        flush=True,
    )

    # Sanitize: a handful of rows in some collections contain stray NaN/Inf
    # (the embedder emitted noise for empty/degenerate articles). One NaN row
    # poisons every query's top-k via NaN-tainted similarities.
    started = time.monotonic()
    bad_rows = 0
    for chunk_start in range(0, total_vectors, 1_000_000):
        chunk_end = min(chunk_start + 1_000_000, total_vectors)
        chunk = embeddings[chunk_start:chunk_end]
        bad_mask = ~np.isfinite(chunk).all(axis=1)
        if bad_mask.any():
            bad_rows += int(bad_mask.sum())
            chunk[bad_mask] = 0
    elapsed = time.monotonic() - started
    print(
        f"sanitized {bad_rows} non-finite rows -> zero vectors in {elapsed:.1f}s",
        flush=True,
    )
    return embeddings


def dense_worker(
    gpu_index: int,
    num_gpus: int,
    embeddings: np.ndarray,
    num_neighbors: int,
    query_tile_rows: int,
    candidate_tile_rows: int,
    scratch_dir: Path,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    total_vectors = embeddings.shape[0]
    stripe_start = (total_vectors * gpu_index) // num_gpus
    stripe_end = (total_vectors * (gpu_index + 1)) // num_gpus
    print(
        f"[gpu{gpu_index} dense] queries [{stripe_start:,}, {stripe_end:,}) "
        f"vs corpus {total_vectors:,}",
        flush=True,
    )
    indices, scores = gt_stripe_dense(
        embeddings_host=embeddings,
        stripe_start=stripe_start,
        stripe_end=stripe_end,
        num_neighbors=num_neighbors,
        query_tile_rows=query_tile_rows,
        candidate_tile_rows=candidate_tile_rows,
        log_prefix=f"[gpu{gpu_index} dense] ",
    )
    scratch_dir.mkdir(parents=True, exist_ok=True)
    write_bin(scratch_dir / f"stripe_{gpu_index:02d}.ibin", indices, dtype="i32")
    write_bin(scratch_dir / f"stripe_{gpu_index:02d}.fbin", scores, dtype="f32")
    print(f"[gpu{gpu_index} dense] DONE -> stripe_{gpu_index:02d}.{{ibin,fbin}}", flush=True)


def maxsim_worker(
    gpu_index: int,
    num_gpus: int,
    token_bank: np.ndarray,
    section_offsets: np.ndarray,
    num_neighbors: int,
    query_tile_sections: int,
    candidate_tile_sections: int,
    scratch_dir: Path,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    total_sections = section_offsets.shape[0] - 1
    stripe_start = (total_sections * gpu_index) // num_gpus
    stripe_end = (total_sections * (gpu_index + 1)) // num_gpus
    print(
        f"[gpu{gpu_index} maxsim] queries [{stripe_start:,}, {stripe_end:,}) "
        f"vs corpus {total_sections:,} sections, {token_bank.shape[0]:,} tokens",
        flush=True,
    )
    indices, scores = gt_stripe_maxsim(
        token_bank_host=token_bank,
        section_offsets_host=section_offsets,
        stripe_start_section=stripe_start,
        stripe_end_section=stripe_end,
        num_neighbors=num_neighbors,
        query_tile_sections=query_tile_sections,
        candidate_tile_sections=candidate_tile_sections,
        log_prefix=f"[gpu{gpu_index} maxsim] ",
    )
    scratch_dir.mkdir(parents=True, exist_ok=True)
    write_bin(scratch_dir / f"stripe_{gpu_index:02d}.ibin", indices, dtype="i32")
    write_bin(scratch_dir / f"stripe_{gpu_index:02d}.fbin", scores, dtype="f32")
    print(f"[gpu{gpu_index} maxsim] DONE -> stripe_{gpu_index:02d}.{{ibin,fbin}}", flush=True)


def gather_outputs(
    scratch_dir: Path,
    model_root: Path,
    suffix: str,
    shards: list[CollectionShard],
    num_gpus: int,
    total_rows: int,
    num_neighbors: int,
) -> None:
    """Slice per-stripe scratch files into per-shard
    `.{suffix}.ground_truth.{ibin,fbin}` files. Each `CollectionShard.row_count`
    is in whatever unit the per-stripe rows were written (articles for dense,
    sections for maxsim) — `gather_outputs` is unit-agnostic.
    """
    bytes_per_row = num_neighbors * 4
    indices_files = [
        open(scratch_dir / f"stripe_{gpu_index:02d}.ibin", "rb")
        for gpu_index in range(num_gpus)
    ]
    scores_files = [
        open(scratch_dir / f"stripe_{gpu_index:02d}.fbin", "rb")
        for gpu_index in range(num_gpus)
    ]
    try:
        stripe_starts = [
            (total_rows * gpu_index) // num_gpus for gpu_index in range(num_gpus + 1)
        ]
        for shard in shards:
            wiki_dir = model_root / shard.wikiname
            wiki_dir.mkdir(parents=True, exist_ok=True)
            indices_path = wiki_dir / f"{shard.stem}.{suffix}.ground_truth.ibin"
            scores_path = wiki_dir / f"{shard.stem}.{suffix}.ground_truth.fbin"
            with (
                open(indices_path, "wb") as out_indices,
                open(scores_path, "wb") as out_scores,
            ):
                out_indices.write(struct.pack("<II", shard.row_count, num_neighbors))
                out_scores.write(struct.pack("<II", shard.row_count, num_neighbors))
                cursor = shard.row_offset
                shard_end = shard.row_offset + shard.row_count
                while cursor < shard_end:
                    stripe_index = next(
                        gpu_index
                        for gpu_index in range(num_gpus)
                        if stripe_starts[gpu_index] <= cursor < stripe_starts[gpu_index + 1]
                    )
                    chunk_end = min(shard_end, stripe_starts[stripe_index + 1])
                    chunk_rows = chunk_end - cursor
                    offset_in_stripe = cursor - stripe_starts[stripe_index]
                    indices_files[stripe_index].seek(8 + offset_in_stripe * bytes_per_row)
                    scores_files[stripe_index].seek(8 + offset_in_stripe * bytes_per_row)
                    out_indices.write(
                        indices_files[stripe_index].read(chunk_rows * bytes_per_row)
                    )
                    out_scores.write(
                        scores_files[stripe_index].read(chunk_rows * bytes_per_row)
                    )
                    cursor = chunk_end
    finally:
        for handle in indices_files + scores_files:
            handle.close()


def run_dense(args, model_root: Path) -> None:
    shards = discover_collection(model_root, args.output_suffix)
    if not shards:
        raise SystemExit(f"no .{args.output_suffix}.f16bin files under {model_root}")
    total_vectors = sum(shard.row_count for shard in shards)
    print(
        f"discovered {len(shards)} shards across "
        f"{len({shard.wikiname for shard in shards})} wikis, "
        f"{total_vectors:,} total rows",
        flush=True,
    )

    # Read dimensions from the first shard's header.
    first_blob = resolve_lfs_pointer(shards[0].path)
    with open(first_blob, "rb") as file:
        _, dimensions = struct.unpack("<II", file.read(8))
    if args.dimensions and args.dimensions != dimensions:
        raise SystemExit(
            f"--dimensions {args.dimensions} != on-disk {dimensions} for {model_root}"
        )

    embeddings = load_dense_collection(model_root, args.output_suffix, dimensions, shards)
    scratch_dir = model_root / f"_ground_truth_scratch_{args.output_suffix}"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    mp_context = mp.get_context("fork")
    workers: list[mp.Process] = []
    for gpu_index in range(args.num_gpus):
        process = mp_context.Process(
            target=dense_worker,
            args=(
                gpu_index,
                args.num_gpus,
                embeddings,
                args.num_neighbors,
                args.query_tile_rows,
                args.candidate_tile_rows,
                scratch_dir,
            ),
        )
        process.start()
        workers.append(process)

    failed = False
    for process in workers:
        process.join()
        if process.exitcode != 0:
            failed = True
            print(
                f"worker pid {process.pid} exited code {process.exitcode}", flush=True
            )
    if failed:
        raise SystemExit("one or more GPU workers failed")

    gather_outputs(
        scratch_dir,
        model_root,
        args.output_suffix,
        shards,
        args.num_gpus,
        total_vectors,
        args.num_neighbors,
    )
    print(
        f"wrote {len(shards)} per-shard "
        f"`.{args.output_suffix}.ground_truth.{{ibin,fbin}}` files under {model_root}",
        flush=True,
    )
    for path in scratch_dir.iterdir():
        path.unlink()
    scratch_dir.rmdir()


def run_maxsim(args, model_root: Path) -> None:
    started = time.monotonic()
    print(f"loading multi-vector corpus under {model_root} ...", flush=True)
    token_bank, section_offsets, shards, dimensions = load_maxsim_corpus(
        model_root, args.output_suffix
    )
    elapsed = time.monotonic() - started
    total_sections = section_offsets.shape[0] - 1
    total_tokens = token_bank.shape[0]
    print(
        f"loaded {total_sections:,} sections, {total_tokens:,} tokens "
        f"({token_bank.nbytes/1e9:.1f} GB) across {len(shards)} shards "
        f"in {elapsed:.1f}s; dim={dimensions}",
        flush=True,
    )

    scratch_dir = model_root / f"_ground_truth_scratch_{args.output_suffix}"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    mp_context = mp.get_context("fork")
    workers: list[mp.Process] = []
    for gpu_index in range(args.num_gpus):
        process = mp_context.Process(
            target=maxsim_worker,
            args=(
                gpu_index,
                args.num_gpus,
                token_bank,
                section_offsets,
                args.num_neighbors,
                args.query_tile_sections,
                args.candidate_tile_sections,
                scratch_dir,
            ),
        )
        process.start()
        workers.append(process)

    failed = False
    for process in workers:
        process.join()
        if process.exitcode != 0:
            failed = True
            print(
                f"worker pid {process.pid} exited code {process.exitcode}", flush=True
            )
    if failed:
        raise SystemExit("one or more GPU workers failed")

    gather_outputs(
        scratch_dir,
        model_root,
        args.output_suffix,
        shards,
        args.num_gpus,
        total_sections,
        args.num_neighbors,
    )
    print(
        f"wrote {len(shards)} per-shard "
        f"`.{args.output_suffix}.ground_truth.{{ibin,fbin}}` files under {model_root}",
        flush=True,
    )
    for path in scratch_dir.iterdir():
        path.unlink()
    scratch_dir.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="dense",
        choices=["dense", "maxsim"],
        help="dense = single vector per row (cosine); maxsim = multi-vector per "
        "section (ColBERT late interaction)",
    )
    parser.add_argument("--output", default="/home/ubuntu/USearchWiki")
    parser.add_argument("--model-subdir", required=True)
    parser.add_argument(
        "--dimensions",
        type=int,
        default=0,
        help="optional sanity check; if 0, read from first shard's header (dense only)",
    )
    parser.add_argument("--output-suffix", default="body", choices=["body", "title"])
    parser.add_argument("--num-neighbors", type=int, default=100)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument(
        "--query-tile-rows",
        type=int,
        default=16384,
        help="dense: rows per query chunk inside the resident stripe",
    )
    parser.add_argument(
        "--candidate-tile-rows",
        type=int,
        default=131072,
        help="dense: rows per candidate tile streamed past the query stripe",
    )
    parser.add_argument(
        "--query-tile-sections",
        type=int,
        default=256,
        help="maxsim: sections per query micro-batch",
    )
    parser.add_argument(
        "--candidate-tile-sections",
        type=int,
        default=65536,
        help="maxsim: sections per streamed candidate tile",
    )
    args = parser.parse_args()

    model_root = Path(args.output) / args.model_subdir
    if not model_root.is_dir():
        raise SystemExit(f"no collection at {model_root}")

    if args.mode == "dense":
        run_dense(args, model_root)
    else:
        run_maxsim(args, model_root)


if __name__ == "__main__":
    main()
