"""Compute exact global k-NN ground truth for an embedding collection.

Streams the entire collection through every GPU as both queries and candidates,
using tiled FP16x FP16 -> FP32 GEMMs with a pre-allocated `out=` buffer. One
collection (model) at a time; partitions queries across GPUs, double-buffers
candidate tile H2D copies, keeps the running top-k on device.

Usage:
  python ground_truth.py --output /path/to/embeddings \\
      --model-subdir qwen3-embedding-0.6b --dimensions 1024 --num-gpus 8
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wikiverse import write_bin


LFS_POINTER_PREFIX = b"version https://git-lfs"


def resolve_lfs_pointer(path: Path) -> Path:
    """If `path` is a Git-LFS pointer file, return the materialized blob in `.git/lfs/objects`.

    Pointer files are tiny ASCII stubs (~133 bytes) with an `oid sha256:<hex>` line.
    Repositories cloned with `GIT_LFS_SKIP_SMUDGE=1` (or with `git lfs fetch` only)
    keep the actual binaries under `.git/lfs/objects/<aa>/<bb>/<oid>` while the
    working tree holds pointers. Reading those blobs in place avoids checking out
    a duplicate copy of the dataset.
    """
    try:
        if path.stat().st_size > 1024:
            return path
    except OSError:
        return path
    with open(path, "rb") as file:
        head = file.read(256)
    if not head.startswith(LFS_POINTER_PREFIX):
        return path
    oid: str | None = None
    for line in head.decode("ascii", errors="ignore").splitlines():
        if line.startswith("oid sha256:"):
            oid = line.split(":", 1)[1].strip()
            break
    if not oid:
        raise ValueError(f"{path}: looks like an LFS pointer but no sha256 oid line")
    # Walk parents of the resolved path (so symlinked working trees still find the
    # original repo's `.git/lfs/objects`).
    for ancestor in path.resolve().parents:
        candidate = ancestor / ".git" / "lfs" / "objects" / oid[:2] / oid[2:4] / oid
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"{path}: LFS pointer references oid {oid} but no .git/lfs/objects/ contains it; "
        f"run `git lfs fetch` or pass --output to a tree that has the blobs"
    )


@dataclass(frozen=True, slots=True)
class CollectionShard:
    wikiname: str
    stem: str
    row_offset: int
    row_count: int


def discover_shards(model_root: Path, suffix: str) -> list[CollectionShard]:
    """Find every {wiki}/{stem}.{suffix}.f16bin under model_root in deterministic order."""
    binaries: list[tuple[str, str, Path, int]] = []
    for wiki_dir in sorted(model_root.iterdir()):
        if not wiki_dir.is_dir():
            continue
        for path in sorted(wiki_dir.glob(f"*.{suffix}.f16bin")):
            stem = path.name[: -len(f".{suffix}.f16bin")]
            blob_path = resolve_lfs_pointer(path)
            with open(blob_path, "rb") as file:
                rows, _columns = struct.unpack("<II", file.read(8))
            binaries.append((wiki_dir.name, stem, path, rows))

    shards: list[CollectionShard] = []
    offset = 0
    for wikiname, stem, _path, row_count in binaries:
        shards.append(
            CollectionShard(
                wikiname=wikiname,
                stem=stem,
                row_offset=offset,
                row_count=row_count,
            )
        )
        offset += row_count
    return shards


def load_collection(
    model_root: Path,
    suffix: str,
    dimensions: int,
    shards: list[CollectionShard],
) -> np.ndarray:
    """Allocate one large host array and stream every shard body into its row range."""
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

    # Sanitize: a handful of rows in some collections contain a stray NaN/Inf
    # (the embedder emitted noise for empty/degenerate articles). Even one
    # NaN row poisons every query's top-k via NaN-tainted similarities.
    # Coerce any non-finite row to a clean zero vector.
    started = time.monotonic()
    bad_rows: list[int] = []
    chunk_rows = 1_000_000
    for chunk_start in range(0, total_vectors, chunk_rows):
        chunk_end = min(chunk_start + chunk_rows, total_vectors)
        chunk = embeddings[chunk_start:chunk_end]
        bad_mask = ~np.isfinite(chunk).all(axis=1)
        if bad_mask.any():
            local_bad = np.where(bad_mask)[0]
            bad_rows.extend((chunk_start + local_bad).tolist())
            chunk[bad_mask] = 0
    elapsed = time.monotonic() - started
    print(
        f"sanitized {len(bad_rows)} non-finite rows -> zero vectors in {elapsed:.1f}s",
        flush=True,
    )
    if bad_rows:
        preview = bad_rows[:8]
        print(f"  bad row indices (first 8): {preview}", flush=True)
    return embeddings


def worker_main(
    gpu_index: int,
    num_gpus: int,
    embeddings: np.ndarray,
    dimensions: int,
    num_neighbors: int,
    query_tile_rows: int,
    candidate_tile_rows: int,
    scratch_dir: Path,
) -> None:
    """One process per GPU. Streams the whole corpus past a resident query stripe."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    import cupy  # imported after CUDA_VISIBLE_DEVICES so this process binds to one GPU
    import torch  # only used for its highly-optimized fused top-k kernel

    total_vectors = embeddings.shape[0]
    stripe_start = (total_vectors * gpu_index) // num_gpus
    stripe_end = (total_vectors * (gpu_index + 1)) // num_gpus
    stripe_size = stripe_end - stripe_start
    keep = num_neighbors + 1  # +1 so we always have headroom to drop the self-match

    print(
        f"[gpu{gpu_index}] queries [{stripe_start:,}, {stripe_end:,}) "
        f"({stripe_size:,} rows), candidate corpus {total_vectors:,} rows",
        flush=True,
    )

    # Resident query stripe on device (~16 GB at 8 GPUs, dim 1024).
    query_stripe = cupy.asarray(embeddings[stripe_start:stripe_end])

    # Pinned host scratch (one per slot) for async H2D of candidate tiles.
    pinned_views: list[np.ndarray] = []
    pinned_holders = []  # keep allocations alive
    for _ in range(2):
        pinned_memory = cupy.cuda.alloc_pinned_memory(
            candidate_tile_rows * dimensions * 2
        )
        pinned_holders.append(pinned_memory)
        view = np.frombuffer(
            pinned_memory, dtype=np.float16, count=candidate_tile_rows * dimensions
        ).reshape(candidate_tile_rows, dimensions)
        pinned_views.append(view)

    # Device candidate buffers (double-buffered).
    candidate_buffers = [
        cupy.empty((candidate_tile_rows, dimensions), dtype=cupy.float16),
        cupy.empty((candidate_tile_rows, dimensions), dtype=cupy.float16),
    ]

    # Pre-allocated FP32 similarity buffer reused as `out=` of every matmul.
    similarity_buffer = cupy.empty(
        (query_tile_rows, candidate_tile_rows), dtype=cupy.float32
    )

    # Running top-k state on device. -inf scores so the first tile populates them.
    topk_scores = cupy.full((stripe_size, keep), -cupy.inf, dtype=cupy.float32)
    topk_indices = cupy.full((stripe_size, keep), -1, dtype=cupy.int32)

    copy_stream = cupy.cuda.Stream(non_blocking=True)
    compute_stream = cupy.cuda.Stream(non_blocking=True)
    copy_done = [
        cupy.cuda.Event(disable_timing=True),
        cupy.cuda.Event(disable_timing=True),
    ]
    compute_done = [
        cupy.cuda.Event(disable_timing=True),
        cupy.cuda.Event(disable_timing=True),
    ]

    candidate_offsets = list(range(0, total_vectors, candidate_tile_rows))

    def stage_tile(slot: int, tile_offset: int) -> int:
        """Stage a candidate tile: pinned scratch then async H2D into candidate_buffers[slot].

        Host-side: waits for any previous H2D from this slot's pinned scratch before
        overwriting it (otherwise the in-flight H2D could read torn data). Device-side:
        waits for the previous compute that read this slot's device buffer.
        """
        count = min(candidate_tile_rows, total_vectors - tile_offset)
        # Pinned scratch reuse: host-side wait on the previous H2D from this slot.
        # (Synchronize on a never-recorded event is a no-op.)
        copy_done[slot].synchronize()
        np.copyto(
            pinned_views[slot][:count], embeddings[tile_offset : tile_offset + count]
        )
        # Device buffer reuse: don't overwrite while compute is still reading it.
        copy_stream.wait_event(compute_done[slot])
        candidate_buffers[slot][:count].set(
            pinned_views[slot][:count], stream=copy_stream
        )
        copy_done[slot].record(copy_stream)
        return count

    # Prime both slots so the steady-state loop has tiles ready on first compute.
    counts = [0, 0]
    for slot in range(min(2, len(candidate_offsets))):
        counts[slot] = stage_tile(slot, candidate_offsets[slot])

    started = time.monotonic()

    for tile_idx, tile_offset in enumerate(candidate_offsets):
        slot = tile_idx % 2
        active_count = counts[slot]
        active_device = candidate_buffers[slot][:active_count]

        # Issue compute for the current tile, waiting for its H2D to complete.
        compute_stream.wait_event(copy_done[slot])

        with compute_stream:
            for query_start in range(0, stripe_size, query_tile_rows):
                query_end = min(query_start + query_tile_rows, stripe_size)
                query_count = query_end - query_start
                similarity_view = similarity_buffer[:query_count, :active_count]

                # Pre-allocated FP32 buffer reused for every tile pair (the `out=` request).
                cupy.matmul(
                    query_stripe[query_start:query_end],
                    active_device.T,
                    out=similarity_view,
                )

                # Top-k for this tile via torch.topk (much faster than
                # cupy.argpartition: ~150x measured at 16K x 64K f32). The DLPack
                # bridge zero-copies the cupy buffer; outputs are still on the
                # same device. When the final tile is shorter than `keep`,
                # take all rows and pad with -inf sentinels.
                if active_count >= keep:
                    similarity_torch = torch.from_dlpack(similarity_view)
                    tile_values_torch, tile_local_torch = torch.topk(
                        similarity_torch, k=keep, dim=1, largest=True, sorted=False
                    )
                    tile_top_scores = cupy.from_dlpack(tile_values_torch)
                    tile_top_indices = cupy.from_dlpack(tile_local_torch).astype(
                        cupy.int32
                    ) + cupy.int32(tile_offset)
                else:
                    pad = keep - active_count
                    sub_global = cupy.arange(
                        active_count, dtype=cupy.int32
                    ) + cupy.int32(tile_offset)
                    tile_top_indices = cupy.concatenate(
                        [
                            cupy.broadcast_to(sub_global, (query_count, active_count)),
                            cupy.full((query_count, pad), -1, dtype=cupy.int32),
                        ],
                        axis=1,
                    )
                    tile_top_scores = cupy.concatenate(
                        [
                            similarity_view,
                            cupy.full(
                                (query_count, pad), -cupy.inf, dtype=cupy.float32
                            ),
                        ],
                        axis=1,
                    )

                # Merge running top-k with this tile's top-k. Combined width is
                # 2 * keep, which is small enough that another torch.topk is the
                # right tool here too.
                running_indices_chunk = topk_indices[query_start:query_end]
                running_scores_chunk = topk_scores[query_start:query_end]
                combined_scores = cupy.concatenate(
                    [running_scores_chunk, tile_top_scores], axis=1
                )
                combined_indices = cupy.concatenate(
                    [running_indices_chunk, tile_top_indices], axis=1
                )
                combined_torch = torch.from_dlpack(combined_scores)
                merge_values_torch, merge_pos_torch = torch.topk(
                    combined_torch, k=keep, dim=1, largest=True, sorted=False
                )
                merge_pos = cupy.from_dlpack(merge_pos_torch)
                topk_scores[query_start:query_end] = cupy.from_dlpack(
                    merge_values_torch
                )
                topk_indices[query_start:query_end] = cupy.take_along_axis(
                    combined_indices, merge_pos, axis=1
                )

            compute_done[slot].record(compute_stream)

        # Now that compute is queued, prefetch tile_idx+2 into this slot.
        # copy_stream waits on compute_done[slot] before overwriting the device buffer,
        # and stage_tile waits host-side on copy_done[slot] before overwriting pinned scratch.
        prefetch_idx = tile_idx + 2
        if prefetch_idx < len(candidate_offsets):
            counts[slot] = stage_tile(slot, candidate_offsets[prefetch_idx])

        if (tile_idx + 1) % 32 == 0 or tile_idx + 1 == len(candidate_offsets):
            compute_stream.synchronize()
            # Reclaim pooled blocks accumulated by intra-tile concat / take ops
            # so pool growth doesn't drift toward the device limit.
            cupy.get_default_memory_pool().free_all_blocks()
            elapsed = time.monotonic() - started
            done_candidates = (tile_idx + 1) * candidate_tile_rows
            millions_per_second = done_candidates / max(elapsed, 1e-3) / 1e6
            print(
                f"[gpu{gpu_index}] tile {tile_idx + 1}/{len(candidate_offsets)} "
                f"elapsed {elapsed:.0f}s ({millions_per_second:.2f}M cand/s)",
                flush=True,
            )

    compute_stream.synchronize()

    # Sort each row by descending score, then drop the self-match in a single shift.
    sorted_order = cupy.argsort(-topk_scores, axis=1)
    sorted_scores = cupy.take_along_axis(topk_scores, sorted_order, axis=1)
    sorted_indices = cupy.take_along_axis(topk_indices, sorted_order, axis=1)

    query_global_ids = cupy.arange(stripe_start, stripe_end, dtype=cupy.int32).reshape(
        -1, 1
    )
    is_self = sorted_indices == query_global_ids
    has_self = cupy.any(is_self, axis=1, keepdims=True)
    self_pos = cupy.argmax(is_self.astype(cupy.int32), axis=1, keepdims=True)

    output_columns = cupy.broadcast_to(
        cupy.arange(num_neighbors, dtype=cupy.int32), (stripe_size, num_neighbors)
    )
    shift_mask = (output_columns >= self_pos) & has_self
    source_columns = output_columns + shift_mask.astype(cupy.int32)
    final_scores = cupy.take_along_axis(sorted_scores, source_columns, axis=1)
    final_indices = cupy.take_along_axis(sorted_indices, source_columns, axis=1)

    scratch_dir.mkdir(parents=True, exist_ok=True)
    indices_path = scratch_dir / f"stripe_{gpu_index:02d}.ibin"
    scores_path = scratch_dir / f"stripe_{gpu_index:02d}.fbin"
    write_bin(indices_path, cupy.asnumpy(final_indices), dtype="i32")
    write_bin(scores_path, cupy.asnumpy(final_scores), dtype="f32")

    elapsed = time.monotonic() - started
    print(
        f"[gpu{gpu_index}] DONE {stripe_size:,} queries in {elapsed:.0f}s "
        f"-> {indices_path.name}, {scores_path.name}",
        flush=True,
    )


def gather_outputs(
    scratch_dir: Path,
    model_root: Path,
    suffix: str,
    shards: list[CollectionShard],
    num_gpus: int,
    total_vectors: int,
    num_neighbors: int,
) -> None:
    """Slice per-stripe scratch files into per-shard `.ground_truth.ibin` and `.ground_truth.fbin`,
    so each shard's ground truth lives next to its source `.f16bin`.

    Each per-stripe scratch is contiguous global rows. A single shard may straddle
    a stripe boundary, so we may pull from up to two stripes per shard.
    """
    bytes_per_row_indices = num_neighbors * 4
    bytes_per_row_scores = num_neighbors * 4
    indices_files = [
        open(scratch_dir / f"stripe_{gpu_index:02d}.ibin", "rb")
        for gpu_index in range(num_gpus)
    ]
    scores_files = [
        open(scratch_dir / f"stripe_{gpu_index:02d}.fbin", "rb")
        for gpu_index in range(num_gpus)
    ]
    try:
        # Stripe boundary table (cumulative row counts).
        stripe_starts = [
            (total_vectors * gpu_index) // num_gpus for gpu_index in range(num_gpus + 1)
        ]
        for shard in shards:
            wiki_dir = model_root / shard.wikiname
            wiki_dir.mkdir(parents=True, exist_ok=True)
            indices_path = wiki_dir / f"{shard.stem}.{suffix}.ground_truth.ibin"
            scores_path = wiki_dir / f"{shard.stem}.{suffix}.ground_truth.fbin"
            with open(indices_path, "wb") as out_indices, open(
                scores_path, "wb"
            ) as out_scores:
                out_indices.write(struct.pack("<II", shard.row_count, num_neighbors))
                out_scores.write(struct.pack("<II", shard.row_count, num_neighbors))
                cursor = shard.row_offset
                shard_end = shard.row_offset + shard.row_count
                while cursor < shard_end:
                    # Find which stripe owns `cursor`.
                    stripe_index = next(
                        gpu_index
                        for gpu_index in range(num_gpus)
                        if stripe_starts[gpu_index]
                        <= cursor
                        < stripe_starts[gpu_index + 1]
                    )
                    chunk_end = min(shard_end, stripe_starts[stripe_index + 1])
                    chunk_rows = chunk_end - cursor
                    offset_in_stripe = cursor - stripe_starts[stripe_index]
                    indices_files[stripe_index].seek(
                        8 + offset_in_stripe * bytes_per_row_indices
                    )
                    scores_files[stripe_index].seek(
                        8 + offset_in_stripe * bytes_per_row_scores
                    )
                    out_indices.write(
                        indices_files[stripe_index].read(
                            chunk_rows * bytes_per_row_indices
                        )
                    )
                    out_scores.write(
                        scores_files[stripe_index].read(
                            chunk_rows * bytes_per_row_scores
                        )
                    )
                    cursor = chunk_end
    finally:
        for handle in indices_files + scores_files:
            handle.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/home/ubuntu/wikiverse-data/embeddings")
    parser.add_argument(
        "--model-subdir",
        default="qwen3-embedding-0.6b",
        help="collection lives at {output}/{model-subdir}/",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=1024,
        help="embedding dimensionality (1024 Qwen3/arctic, 768 nomic, 4096 e5-mistral)",
    )
    parser.add_argument("--output-suffix", default="body", choices=["body", "title"])
    parser.add_argument("--num-neighbors", type=int, default=100)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument(
        "--query-tile-rows",
        type=int,
        default=16384,
        help="rows per query chunk inside the resident stripe",
    )
    parser.add_argument(
        "--candidate-tile-rows",
        type=int,
        default=131072,
        help="rows per candidate tile streamed past the query stripe",
    )
    args = parser.parse_args()

    model_root = Path(args.output) / args.model_subdir
    if not model_root.is_dir():
        raise SystemExit(f"no collection at {model_root}")

    shards = discover_shards(model_root, args.output_suffix)
    if not shards:
        raise SystemExit(f"no .{args.output_suffix}.f16bin files under {model_root}")
    total_vectors = sum(shard.row_count for shard in shards)
    print(
        f"discovered {len(shards)} shards across "
        f"{len({shard.wikiname for shard in shards})} wikis, "
        f"{total_vectors:,} total rows",
        flush=True,
    )

    embeddings = load_collection(
        model_root, args.output_suffix, args.dimensions, shards
    )

    scratch_dir = model_root / f"_ground_truth_scratch_{args.output_suffix}"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    mp_context = mp.get_context("fork")
    workers: list[mp.Process] = []
    for gpu_index in range(args.num_gpus):
        process = mp_context.Process(
            target=worker_main,
            args=(
                gpu_index,
                args.num_gpus,
                embeddings,
                args.dimensions,
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
                f"worker pid {process.pid} exited with code {process.exitcode}",
                flush=True,
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
        f"`.{args.output_suffix}.ground_truth.ibin` + `.{args.output_suffix}.ground_truth.fbin` "
        f"files under {model_root}",
        flush=True,
    )

    for path in scratch_dir.iterdir():
        path.unlink()
    scratch_dir.rmdir()


if __name__ == "__main__":
    main()
