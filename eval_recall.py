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

from usearchwiki import (  # noqa: E402
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


def metrics_at_k(
    expected_keys: np.ndarray,
    actual_keys: np.ndarray,
    k_recall: int,
    k_ndcg: int,
) -> tuple[float, float]:
    """Compute strict Recall@k_recall and binary NDCG@k_ndcg.

    `expected_keys` is the exact top-k_max ground truth (descending
    similarity), `actual_keys` is the predicted top-k_max from the index
    (self-match already removed). Both arrays are
    `(n_queries, k_max)` with `k_max >= max(k_recall, k_ndcg)`.

    Strict recall: predicted top-k_recall key counts iff it appears in GT
    *top-k_recall*. Standard ANN-Benchmarks definition.

    Binary NDCG: predicted top-k_ndcg key counts iff it appears in GT
    *top-k_ndcg*. Both rankings are graded by their position in their
    respective top-lists, so a predicted #1 that matches GT #50 still
    contributes 1 / log2(2) at rank 1 in DCG.
    """
    # Recall@k_recall: small bool matrix from k_recall slices on both sides.
    rec_actual = actual_keys[:, :k_recall]
    rec_expected = expected_keys[:, :k_recall]
    membership_recall = (rec_actual[:, :, None] == rec_expected[:, None, :]).any(axis=2)
    recall = float(membership_recall.sum(axis=1).mean()) / k_recall

    # NDCG@k_ndcg: bigger bool matrix from k_ndcg slices.
    ndcg_actual = actual_keys[:, :k_ndcg]
    ndcg_expected = expected_keys[:, :k_ndcg]
    membership_ndcg = (
        ndcg_actual[:, :, None] == ndcg_expected[:, None, :]
    ).any(axis=2)
    discount = 1.0 / np.log2(np.arange(2, k_ndcg + 2))
    dcg = (membership_ndcg * discount).sum(axis=1)
    idcg = float(discount.sum())  # |GT| >= k_ndcg by construction
    ndcg = float((dcg / idcg).mean())
    return recall, ndcg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/home/ubuntu/WikiVerse")
    parser.add_argument("--model-subdir", required=True)
    parser.add_argument("--output-suffix", default="body", choices=["body", "title"])
    parser.add_argument("--index", type=Path, default=None)
    parser.add_argument("--num-queries", type=int, default=10000)
    parser.add_argument(
        "--k-recall",
        type=int,
        default=10,
        help="cutoff for Recall@k",
    )
    parser.add_argument(
        "--k-ndcg",
        type=int,
        default=100,
        help="cutoff for NDCG@k; also drives how many GT neighbors are loaded "
        "and how many results we ask the index for (k_ndcg + 1 to drop self)",
    )
    parser.add_argument(
        "--ef-search",
        default="16,32,64,128,256,512,1024",
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
        model_root, args.output_suffix, shards, query_ids, args.k_ndcg
    )
    print(f"  loaded in {time.monotonic()-started:.1f}s", flush=True)

    ef_values = [int(x) for x in args.ef_search.split(",") if x.strip()]
    print(
        f"sweeping ef_search over {ef_values} "
        f"(recall@{args.k_recall}, ndcg@{args.k_ndcg}) ...",
        flush=True,
    )
    print(
        f"{'ef_search':>10}  "
        f"{'recall@'+str(args.k_recall):>12}  {'recall q/s':>12}  "
        f"{'ndcg@'+str(args.k_ndcg):>12}  {'ndcg q/s':>12}"
    )

    # Two search calls per ef: one with count=k_recall+1 to get a meaningful
    # recall@k_recall curve at the requested ef, and one with count=k_ndcg+1
    # for NDCG. USearch coerces the internal expansion to >= count, so a
    # single shared count=k_ndcg+1 would flatten the recall@k_recall sweep
    # at low ef (effective ef becomes k_ndcg+1 regardless).
    def search_top(count: int) -> tuple[np.ndarray, float]:
        started = time.monotonic()
        results = index.search(query_vectors, count=count, threads=args.threads)
        elapsed = time.monotonic() - started
        raw_keys = np.asarray(results.keys, dtype=np.int64)
        actual = np.empty((args.num_queries, count - 1), dtype=np.int64)
        target = count - 1
        for row in range(args.num_queries):
            without_self = raw_keys[row][raw_keys[row] != query_ids[row]][:target]
            if without_self.shape[0] < target:
                actual[row] = -1
                actual[row, : without_self.shape[0]] = without_self
            else:
                actual[row] = without_self
        return actual, elapsed

    expected_recall = expected_keys[:, : args.k_recall]
    expected_ndcg = expected_keys[:, : args.k_ndcg]
    discount = 1.0 / np.log2(np.arange(2, args.k_ndcg + 2))
    idcg = float(discount.sum())

    for ef in ef_values:
        index.expansion_search = ef
        # --- recall sweep (small count) ---
        actual_recall, elapsed_recall = search_top(args.k_recall + 1)
        membership_recall = (
            actual_recall[:, :, None] == expected_recall[:, None, :]
        ).any(axis=2)
        recall = float(membership_recall.sum(axis=1).mean()) / args.k_recall
        rate_recall = args.num_queries / max(elapsed_recall, 1e-3)
        # --- ndcg sweep (large count) ---
        actual_ndcg, elapsed_ndcg = search_top(args.k_ndcg + 1)
        membership_ndcg = (
            actual_ndcg[:, :, None] == expected_ndcg[:, None, :]
        ).any(axis=2)
        dcg = (membership_ndcg * discount).sum(axis=1)
        ndcg = float((dcg / idcg).mean())
        rate_ndcg = args.num_queries / max(elapsed_ndcg, 1e-3)
        print(
            f"{ef:>10}  "
            f"{recall*100:>11.4f}%  {rate_recall:>12,.0f}  "
            f"{ndcg*100:>11.4f}%  {rate_ndcg:>12,.0f}"
        )


if __name__ == "__main__":
    main()
