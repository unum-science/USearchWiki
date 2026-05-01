"""Measure self-recall@k of a USearch index against the precomputed exact
ground-truth files (`{wiki}/{stem}.body.ground_truth.ibin`).

Samples N article IDs uniformly across the canonical shard walk, queries the
index with their stored vectors (memory-mapped from the same `.f16bin` files),
and compares the returned top-k keys against the exact top-k from the ground
truth. Reports mean recall@k for one or more `ef_search` settings — the
standard recall-vs-speed sweep.

Usage:
  python eval_recall.py \\
      --output /home/ubuntu/WikiVerse \\
      --model-subdir qwen3-embedding-0.6b \\
      --output-suffix body \\
      --index /home/ubuntu/WikiVerse/qwen3-embedding-0.6b/body.usearch \\
      --num-queries 10000 --k 10 \\
      --ef-search 64,128,256,512
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from wikiverse import (  # noqa: E402
    CollectionShard,
    discover_collection,
    resolve_lfs_pointer,
)


def memmap_shard_vectors(shard: CollectionShard, dimensions: int) -> np.ndarray:
    blob = resolve_lfs_pointer(shard.path)
    return np.memmap(
        blob,
        dtype=np.float16,
        mode="r",
        offset=8,
        shape=(shard.row_count, dimensions),
    )


def gather_query_vectors(
    shards: list[CollectionShard],
    dimensions: int,
    query_global_ids: np.ndarray,
) -> np.ndarray:
    """Return a `(len(query_global_ids), dimensions)` FP16 array of the
    embeddings for the given global IDs, drawn via memmap from the canonical
    shard walk.
    """
    out = np.empty((len(query_global_ids), dimensions), dtype=np.float16)
    # Index of each query within the per-shard offsets — binary search.
    shard_starts = np.array([s.row_offset for s in shards], dtype=np.int64)
    shard_indices = np.searchsorted(shard_starts, query_global_ids, side="right") - 1
    sort_order = np.argsort(shard_indices, kind="stable")
    sorted_query_indices = shard_indices[sort_order]
    sorted_global_ids = query_global_ids[sort_order]

    cursor = 0
    while cursor < len(sort_order):
        shard_index = int(sorted_query_indices[cursor])
        end = cursor
        while end < len(sort_order) and sorted_query_indices[end] == shard_index:
            end += 1
        shard = shards[shard_index]
        local_rows = sorted_global_ids[cursor:end] - shard.row_offset
        memmap = memmap_shard_vectors(shard, dimensions)
        out[sort_order[cursor:end]] = np.asarray(memmap[local_rows])
        cursor = end
    return out


def gather_ground_truth(
    model_root: Path,
    suffix: str,
    shards: list[CollectionShard],
    query_global_ids: np.ndarray,
    k: int,
) -> np.ndarray:
    """Return `(len(query_global_ids), k)` int32 array of exact top-k indices
    pulled from per-shard `.{suffix}.ground_truth.ibin` files. Assumes the GT
    was stored with at least `k` neighbors per row."""
    out = np.empty((len(query_global_ids), k), dtype=np.int32)
    shard_starts = np.array([s.row_offset for s in shards], dtype=np.int64)
    shard_indices = np.searchsorted(shard_starts, query_global_ids, side="right") - 1
    sort_order = np.argsort(shard_indices, kind="stable")
    sorted_query_indices = shard_indices[sort_order]
    sorted_global_ids = query_global_ids[sort_order]
    cursor = 0
    while cursor < len(sort_order):
        shard_index = int(sorted_query_indices[cursor])
        end = cursor
        while end < len(sort_order) and sorted_query_indices[end] == shard_index:
            end += 1
        shard = shards[shard_index]
        gt_path = (
            model_root / shard.wikiname / f"{shard.stem}.{suffix}.ground_truth.ibin"
        )
        gt_blob = resolve_lfs_pointer(gt_path)
        with open(gt_blob, "rb") as file:
            rows, gt_k = struct.unpack("<II", file.read(8))
        if k > gt_k:
            raise SystemExit(
                f"requested k={k} > stored ground-truth k={gt_k} in {gt_path}"
            )
        gt_memmap = np.memmap(
            gt_blob, dtype=np.int32, mode="r", offset=8, shape=(rows, gt_k)
        )
        local_rows = sorted_global_ids[cursor:end] - shard.row_offset
        out[sort_order[cursor:end]] = np.asarray(gt_memmap[local_rows, :k])
        cursor = end
    return out


def measure_recall(
    expected: np.ndarray,
    actual: np.ndarray,
    k: int,
) -> float:
    """`expected` and `actual` are both `(num_queries, k)` int arrays."""
    matches = 0
    for row in range(expected.shape[0]):
        matches += len(set(expected[row].tolist()) & set(actual[row].tolist()))
    return matches / (expected.shape[0] * k)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/home/ubuntu/WikiVerse")
    parser.add_argument("--model-subdir", required=True)
    parser.add_argument("--output-suffix", default="body", choices=["body", "title"])
    parser.add_argument("--index", type=Path, default=None)
    parser.add_argument("--num-queries", type=int, default=10000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--ef-search",
        default="64,128,256,512",
        help="comma-separated efSearch values to sweep",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=64,
        help="USearch search thread count",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from usearch.index import Index

    model_root = Path(args.output) / args.model_subdir
    index_path = (
        args.index
        if args.index is not None
        else model_root / f"{args.output_suffix}.usearch"
    )
    if not index_path.is_file():
        raise SystemExit(f"index not found at {index_path}")

    print(f"loading index {index_path} ...", flush=True)
    started = time.monotonic()
    index = Index.restore(str(index_path))
    print(
        f"  loaded {len(index):,} vectors in {time.monotonic()-started:.1f}s",
        flush=True,
    )

    print(f"discovering shards under {model_root} ...", flush=True)
    shards = discover_collection(model_root, args.output_suffix)
    total_vectors = sum(s.row_count for s in shards)
    if total_vectors != len(index):
        print(
            f"  WARN: index size {len(index):,} != collection size {total_vectors:,}",
            flush=True,
        )
    # Read first shard header for dimensions
    first_blob = resolve_lfs_pointer(shards[0].path)
    with open(first_blob, "rb") as file:
        _, dimensions = struct.unpack("<II", file.read(8))
    print(f"  {total_vectors:,} vectors x {dimensions}d", flush=True)

    rng = np.random.default_rng(args.seed)
    query_ids = np.sort(
        rng.choice(total_vectors, size=args.num_queries, replace=False)
    ).astype(np.int64)
    print(f"sampled {args.num_queries:,} query IDs", flush=True)

    print("loading query vectors and exact ground truth ...", flush=True)
    started = time.monotonic()
    query_vectors = gather_query_vectors(shards, dimensions, query_ids)
    expected_keys = gather_ground_truth(
        model_root, args.output_suffix, shards, query_ids, args.k
    )
    print(f"  loaded in {time.monotonic()-started:.1f}s", flush=True)

    ef_values = [int(x) for x in args.ef_search.split(",") if x.strip()]
    print(f"sweeping ef_search over {ef_values} with k={args.k} ...", flush=True)
    print(f"{'ef_search':>10}  {'recall@k':>10}  {'queries/s':>12}")
    for ef in ef_values:
        index.expansion_search = ef
        started = time.monotonic()
        results = index.search(query_vectors, count=args.k, threads=args.threads)
        elapsed = time.monotonic() - started
        actual_keys = np.asarray(results.keys, dtype=np.int64)
        recall = measure_recall(expected_keys, actual_keys, args.k)
        rate = args.num_queries / max(elapsed, 1e-3)
        print(f"{ef:>10}  {recall*100:>9.4f}%  {rate:>12,.0f}")


if __name__ == "__main__":
    main()
