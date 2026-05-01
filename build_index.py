"""Build a USearch index from per-shard `.f16bin` files in canonical order.

For dense article-level collections (one vector per article, multi=False).
The ColBERT section-level case (multi=True, vectors keyed by article) will
be a separate addition once the section embeddings finish computing.

Memory-maps the LFS-resolved `.f16bin` blobs so the OS pages vectors in
lazily — keeps RSS bounded when running multiple builds in parallel.

Inspired by ashvardanian/RetriEval's USearch wrapper:
  - reserve capacity up front
  - parallel `add()` with a fixed thread count
  - cosine metric, f16 quantization (matching storage dtype)
  - HNSW hyperparameters connectivity / expansion_add
  - ef_search is a *query-time* knob, not set here
"""

from __future__ import annotations

import argparse
import os
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


def memmap_shard(shard: CollectionShard, dimensions: int) -> np.ndarray:
    """Memory-map a `.f16bin` shard skipping its 8-byte header.

    Returns a read-only `(rows, dimensions)` float16 view. The OS pages in
    only the bytes actually accessed during `index.add()`.
    """
    blob = resolve_lfs_pointer(shard.path)
    return np.memmap(
        blob,
        dtype=np.float16,
        mode="r",
        offset=8,
        shape=(shard.row_count, dimensions),
    )


def add_shards(
    index,
    shards: list[CollectionShard],
    dimensions: int,
    threads: int,
    log_every: int,
) -> int:
    """Stream every shard's vectors into the index. Keys are sequential
    global row IDs assigned in shard-walk order (== `shard.row_offset + i`).
    """
    cumulative_rows = 0
    started = time.monotonic()
    bytes_added_since_log = 0
    last_log_at = started
    for shard_index, shard in enumerate(shards):
        vectors = memmap_shard(shard, dimensions)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="/home/ubuntu/WikiVerse",
        help="root directory holding {model-subdir}/{wiki}/*.f16bin",
    )
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
        help="HNSW M, neighbors per node (16 is the typical floor for ANN)",
    )
    parser.add_argument(
        "--expansion-add",
        type=int,
        default=256,
        help="HNSW efConstruction; bumped from the default 128 to chase >=99% recall@10",
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
    args = parser.parse_args()

    from usearch.index import Index  # local import: heavy dependency

    model_root = Path(args.output) / args.model_subdir
    print(f"discovering shards under {model_root} ...", flush=True)
    started = time.monotonic()
    shards = discover_collection(model_root, args.output_suffix)
    if not shards:
        raise SystemExit(f"no .{args.output_suffix}.f16bin shards under {model_root}")
    # Read dimensions from the first shard's header. (Within a model the
    # collection is consistent by construction; if it weren't, `index.add`
    # would raise on the first mismatched shard anyway.)
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

    output_index_path = (
        args.output_index
        if args.output_index is not None
        else model_root / f"{args.output_suffix}.usearch"
    )

    print(
        f"opening USearch index "
        f"(dim={dimensions}, metric={args.metric}, dtype={args.dtype}, "
        f"M={args.connectivity}, ef_add={args.expansion_add}, "
        f"multi=False, threads={args.threads})",
        flush=True,
    )
    index = Index(
        ndim=dimensions,
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
        dimensions=dimensions,
        threads=args.threads,
        log_every=args.log_every,
    )
    elapsed_build = time.monotonic() - started
    rate = added / max(elapsed_build, 1e-3)
    print(
        f"added {added:,} vectors in {elapsed_build:.0f}s "
        f"({rate:,.0f} vec/s), index size now {len(index):,}",
        flush=True,
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


if __name__ == "__main__":
    main()
