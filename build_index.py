"""Build a USearch index from per-shard `.f16bin` files in canonical order.

Two modes:
  - default: build the index, save to disk as `{suffix}.usearch`.
  - `--no-save --ef-search-sweep ef1,ef2,...`: build the index in memory,
    evaluate Recall@k_recall and NDCG@k_ndcg across the ef sweep, append a
    row per ef to `--stats-jsonl`, then drop the index without saving. This
    is the Matryoshka-style quality-vs-width sweep — useful when you want
    the *numbers* but not the artefacts.

Memory-maps the LFS-resolved `.f16bin` blobs so the OS pages vectors in
lazily — keeps RSS bounded when running multiple builds in parallel.
"""

from __future__ import annotations

import argparse
import json
import os
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


def memmap_shard(
    shard: CollectionShard,
    dimensions_full: int,
    dimensions_target: int | None = None,
) -> np.ndarray:
    """Memory-map a `.f16bin` shard skipping its 8-byte header.

    When `dimensions_target == dimensions_full` the result is a read-only
    memmap'd view (zero-copy). When `dimensions_target < dimensions_full`
    the leading `dimensions_target` columns are sliced and L2-renormalized
    in FP32 before being cast back to FP16 — the standard Matryoshka
    truncation recipe.
    """
    blob = resolve_lfs_pointer(shard.path)
    full = np.memmap(
        blob,
        dtype=np.float16,
        mode="r",
        offset=8,
        shape=(shard.row_count, dimensions_full),
    )
    if dimensions_target is None or dimensions_target == dimensions_full:
        return full
    sliced = np.asarray(full[:, :dimensions_target], dtype=np.float32)
    norms = np.linalg.norm(sliced, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (sliced / norms).astype(np.float16)


def add_shards(
    index,
    shards: list[CollectionShard],
    dimensions_full: int,
    dimensions_target: int,
    threads: int,
    log_every: int,
) -> int:
    """Stream every shard's vectors into the index. Keys are sequential global
    row IDs assigned in shard-walk order (`shard.row_offset + i`).
    """
    cumulative_rows = 0
    started = time.monotonic()
    bytes_added_since_log = 0
    last_log_at = started
    for shard_index, shard in enumerate(shards):
        vectors = memmap_shard(shard, dimensions_full, dimensions_target)
        keys = np.arange(
            shard.row_offset, shard.row_offset + shard.row_count, dtype=np.uint64
        )
        index.add(keys=keys, vectors=vectors, threads=threads)
        cumulative_rows += shard.row_count
        bytes_added_since_log += vectors.nbytes
        if (shard_index + 1) % log_every == 0 or shard_index == len(shards) - 1:
            now = time.monotonic()
            elapsed = now - started
            interval = now - last_log_at
            rate = cumulative_rows / max(elapsed, 1e-3)
            interval_mb = bytes_added_since_log / 1e6 / max(interval, 1e-3)
            print(
                f"  shard {shard_index + 1}/{len(shards)} "
                f"({shard.wikiname}/{shard.stem}): "
                f"{cumulative_rows:,} vectors total, "
                f"{rate:,.0f} vec/s avg, {interval_mb:,.0f} MB/s recent",
                flush=True,
            )
            last_log_at = now
            bytes_added_since_log = 0
    return cumulative_rows


def evaluate_and_log(
    index,
    args,
    shards: list[CollectionShard],
    model_root: Path,
    dimensions_full: int,
    target_dim: int,
    total_vectors: int,
    build_seconds: float,
) -> None:
    """Run a Recall@k_recall + NDCG@k_ndcg sweep across the ef_search values
    and append one JSONL row per ef to `args.stats_jsonl`. Uses the eval
    helpers from `eval_recall` (gather queries, gather GT, metrics_at_k).
    """
    from eval_recall import gather_ground_truth, gather_query_vectors

    rng = np.random.default_rng(args.seed)
    query_ids = np.sort(
        rng.choice(total_vectors, size=args.num_queries, replace=False)
    ).astype(np.int64)
    query_vectors_full = gather_query_vectors(shards, dimensions_full, query_ids)
    if target_dim != dimensions_full:
        query_vectors = np.asarray(query_vectors_full[:, :target_dim], dtype=np.float32)
        norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        query_vectors = (query_vectors / norms).astype(np.float16)
    else:
        query_vectors = query_vectors_full

    expected_keys = gather_ground_truth(
        model_root, args.output_suffix, shards, query_ids, args.k_ndcg
    )

    ef_values = [int(x) for x in args.ef_search_sweep.split(",") if x.strip()]
    print(
        f"sweeping ef_search over {ef_values} "
        f"(recall@{args.k_recall}, ndcg@{args.k_ndcg}) ...",
        flush=True,
    )
    print(
        f"{'ef_search':>10}  {'recall@'+str(args.k_recall):>12}  "
        f"{'ndcg@'+str(args.k_ndcg):>12}  {'recall q/s':>12}  {'ndcg q/s':>12}",
        flush=True,
    )

    args.stats_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    build_rate = total_vectors / max(build_seconds, 1e-3)
    index_bytes_estimate = (
        total_vectors * target_dim * 2
        + total_vectors * args.connectivity * 4 * 2
    )

    def search_top(count: int) -> tuple[np.ndarray, float]:
        started = time.monotonic()
        results = index.search(query_vectors, count=count, threads=args.threads)
        elapsed = time.monotonic() - started
        raw_keys = np.asarray(results.keys, dtype=np.int64)
        target = count - 1
        actual = np.empty((args.num_queries, target), dtype=np.int64)
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
        actual_recall, elapsed_recall = search_top(args.k_recall + 1)
        membership_recall = (
            actual_recall[:, :, None] == expected_recall[:, None, :]
        ).any(axis=2)
        recall = float(membership_recall.sum(axis=1).mean()) / args.k_recall

        actual_ndcg, elapsed_ndcg = search_top(args.k_ndcg + 1)
        membership_ndcg = (
            actual_ndcg[:, :, None] == expected_ndcg[:, None, :]
        ).any(axis=2)
        dcg = (membership_ndcg * discount).sum(axis=1)
        ndcg = float((dcg / idcg).mean())

        rate_recall = args.num_queries / max(elapsed_recall, 1e-3)
        rate_ndcg = args.num_queries / max(elapsed_ndcg, 1e-3)
        print(
            f"{ef:>10}  {recall*100:>11.4f}%  {ndcg*100:>11.4f}%  "
            f"{rate_recall:>12,.0f}  {rate_ndcg:>12,.0f}",
            flush=True,
        )
        rows.append(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "model_subdir": args.model_subdir,
                "output_suffix": args.output_suffix,
                "dimensions_full": dimensions_full,
                "dimensions_indexed": target_dim,
                "connectivity": args.connectivity,
                "expansion_add": args.expansion_add,
                "expansion_search": ef,
                "metric": args.metric,
                "dtype": args.dtype,
                "total_vectors": int(total_vectors),
                "num_queries": int(args.num_queries),
                "k_recall": int(args.k_recall),
                "k_ndcg": int(args.k_ndcg),
                "recall": recall,
                "ndcg": ndcg,
                "queries_per_second_recall": rate_recall,
                "queries_per_second_ndcg": rate_ndcg,
                "build_seconds": build_seconds,
                "build_vec_per_second": build_rate,
                "index_bytes_estimate": int(index_bytes_estimate),
                "build_threads": int(args.threads),
            }
        )

    with open(args.stats_jsonl, "a") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")
    print(
        f"appended {len(rows)} rows to {args.stats_jsonl}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/home/ubuntu/USearchWiki")
    parser.add_argument(
        "--model-subdir",
        required=True,
        help="e.g. qwen3-embedding-0.6b, nomic-embed-text-v1.5, snowflake-arctic-embed-l-v2.0",
    )
    parser.add_argument(
        "--output-suffix",
        default="body",
        choices=["body", "title"],
        help="which embedding file flavor to index",
    )
    parser.add_argument(
        "--output-index",
        type=Path,
        default=None,
        help="destination .usearch file (defaults to {output}/{model-subdir}/{suffix}.usearch)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=os.cpu_count() or 1,
        help="parallel insertion threads (default: all logical cores)",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        default=16,
        help="HNSW M, neighbors per node",
    )
    parser.add_argument(
        "--expansion-add",
        type=int,
        default=256,
        help="HNSW efConstruction; bumped from 128 to chase >=99% recall@10",
    )
    parser.add_argument(
        "--metric",
        default="cos",
        choices=["cos", "ip", "l2sq"],
        help="similarity metric; cos is right for L2-normalized embeddings",
    )
    parser.add_argument(
        "--dtype",
        default="f16",
        help="index quantization dtype; f16 matches the on-disk format",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="print a progress line every N shards",
    )
    parser.add_argument(
        "--truncate-dim",
        type=int,
        default=0,
        help="truncate stored embeddings to the leading N dimensions (Matryoshka). "
        "0 means no truncation.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="build the index in memory and drop it; do not write `{suffix}.usearch`. "
        "Useful for the Matryoshka quality sweep where indexes are evaluated then thrown away.",
    )
    parser.add_argument(
        "--ef-search-sweep",
        default="",
        help="comma-separated ef_search values to evaluate post-build. When set, runs "
        "Recall@k_recall + NDCG@k_ndcg and appends a JSONL row per ef to --stats-jsonl.",
    )
    parser.add_argument("--num-queries", type=int, default=10000)
    parser.add_argument("--k-recall", type=int, default=10)
    parser.add_argument("--k-ndcg", type=int, default=100)
    parser.add_argument(
        "--stats-jsonl",
        type=Path,
        default=Path("/home/ubuntu/wikiverse-data/logs/index-stats.jsonl"),
        help="JSONL file to append per-ef sweep rows to (only used with --ef-search-sweep)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from usearch.index import Index  # local import: heavy dependency

    model_root = Path(args.output) / args.model_subdir
    print(f"discovering shards under {model_root} ...", flush=True)
    started = time.monotonic()
    shards = discover_collection(model_root, args.output_suffix)
    if not shards:
        raise SystemExit(f"no .{args.output_suffix}.f16bin shards under {model_root}")
    first_blob = resolve_lfs_pointer(shards[0].path)
    with open(first_blob, "rb") as file:
        _, dimensions = struct.unpack("<II", file.read(8))
    total_vectors = sum(s.row_count for s in shards)
    elapsed = time.monotonic() - started
    print(
        f"  {len(shards)} shards across "
        f"{len({s.wikiname for s in shards})} wikis, "
        f"{total_vectors:,} vectors x {dimensions}d in {elapsed:.1f}s",
        flush=True,
    )

    if args.truncate_dim and args.truncate_dim != dimensions:
        if args.truncate_dim > dimensions:
            raise SystemExit(
                f"--truncate-dim {args.truncate_dim} > native {dimensions}"
            )
        target_dim = args.truncate_dim
    else:
        target_dim = dimensions

    print(
        f"opening USearch index "
        f"(dim={target_dim}, metric={args.metric}, dtype={args.dtype}, "
        f"M={args.connectivity}, ef_add={args.expansion_add}, "
        f"multi=False, threads={args.threads}, "
        f"truncated_from={dimensions if target_dim != dimensions else None})",
        flush=True,
    )
    index = Index(
        ndim=target_dim,
        metric=args.metric,
        dtype=args.dtype,
        connectivity=args.connectivity,
        expansion_add=args.expansion_add,
        multi=False,
    )

    print("streaming shards into index ...", flush=True)
    started = time.monotonic()
    added = add_shards(
        index=index,
        shards=shards,
        dimensions_full=dimensions,
        dimensions_target=target_dim,
        threads=args.threads,
        log_every=args.log_every,
    )
    build_seconds = time.monotonic() - started
    rate = added / max(build_seconds, 1e-3)
    print(
        f"added {added:,} vectors in {build_seconds:.0f}s "
        f"({rate:,.0f} vec/s), index size now {len(index):,}",
        flush=True,
    )

    if args.ef_search_sweep.strip():
        evaluate_and_log(
            index=index,
            args=args,
            shards=shards,
            model_root=model_root,
            dimensions_full=dimensions,
            target_dim=target_dim,
            total_vectors=total_vectors,
            build_seconds=build_seconds,
        )

    if not args.no_save:
        output_index_path = (
            args.output_index
            if args.output_index is not None
            else model_root / f"{args.output_suffix}.usearch"
        )
        output_index_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        index.save(str(output_index_path))
        elapsed_save = time.monotonic() - started
        file_size_gb = output_index_path.stat().st_size / 1e9
        print(
            f"saved {output_index_path} ({file_size_gb:.2f} GB) in {elapsed_save:.0f}s",
            flush=True,
        )
    else:
        print("--no-save set; dropping index without writing to disk", flush=True)


if __name__ == "__main__":
    main()
