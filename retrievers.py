"""Brute-force retrievers + shared GPU kernels.

Two retriever families share one streaming-top-k template:
  - DenseRetriever: cosine top-k over an `(N, dim)` FP16 corpus.
  - MaxSimRetriever: ColBERT-style late interaction over a multi-vector
    corpus (`(T_total, dim)` token bank + `(N+1,)` section offsets).

Both compute an `(Q, M)` score matrix per candidate tile, merge it into the
running top-k, and repeat. They differ only in the per-tile score kernel:

  - Dense: a single FP16 matmul into a pre-allocated FP32 `out=` buffer.
  - MaxSim: matmul + segment-max along ragged doc-token offsets +
    segment-sum along ragged query-token offsets, via two small RawKernels.

The same streaming loop is exposed two ways:
  - `gt_stripe_*` functions: per-GPU stripe workers used by `ground_truth.py`
    for the all-vs-all corpus sweep (host-resident corpus, double-buffered
    H2D for candidate tiles, query stripe resident on-device).
  - `DenseRetriever` / `MaxSimRetriever` classes: consumer-facing search,
    corpus loaded once at construction and held resident on a single GPU.
    `search()` becomes a single `gt_stripe_*` call with the resident corpus
    used as the candidate stream.
"""

from __future__ import annotations

import struct
import time
from pathlib import Path

import numpy as np

from usearchwiki import (
    CollectionShard,
    discover_collection,
    resolve_lfs_pointer,
)


# Two ragged-segment reductions for MaxSim. Each thread handles one (row,
# segment) cell; the inner loop is bounded by the segment width (median 3,
# p99 ~16 for FineWiki sections, so a single warp easily covers the worst
# rows without divergence pain).
_SEGMENT_MAX_SRC = r"""
extern "C" __global__ void segment_max_2d(
    const float* __restrict__ values,
    const int* __restrict__ offsets,
    float* __restrict__ out,
    int rows, int n_segments, int row_stride, int out_stride
) {
    int seg = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows || seg >= n_segments) return;
    int start = offsets[seg];
    int end   = offsets[seg + 1];
    const float* row_ptr = values + row * row_stride;
    float best = -3.4e38f;
    for (int t = start; t < end; ++t) {
        float v = row_ptr[t];
        if (v > best) best = v;
    }
    out[row * out_stride + seg] = best;
}
"""

_SEGMENT_SUM_SRC = r"""
extern "C" __global__ void segment_sum_2d(
    const float* __restrict__ values,
    const int* __restrict__ offsets,
    float* __restrict__ out,
    int n_segments, int n_cols, int row_stride, int out_stride
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int seg = blockIdx.y * blockDim.y + threadIdx.y;
    if (seg >= n_segments || col >= n_cols) return;
    int start = offsets[seg];
    int end   = offsets[seg + 1];
    float total = 0.0f;
    for (int t = start; t < end; ++t) {
        total += values[t * row_stride + col];
    }
    out[seg * out_stride + col] = total;
}
"""

_SEGMENT_MAX_KERNEL = None
_SEGMENT_SUM_KERNEL = None


def _segment_kernels():
    global _SEGMENT_MAX_KERNEL, _SEGMENT_SUM_KERNEL
    if _SEGMENT_MAX_KERNEL is None:
        import cupy

        _SEGMENT_MAX_KERNEL = cupy.RawKernel(_SEGMENT_MAX_SRC, "segment_max_2d")
        _SEGMENT_SUM_KERNEL = cupy.RawKernel(_SEGMENT_SUM_SRC, "segment_sum_2d")
    return _SEGMENT_MAX_KERNEL, _SEGMENT_SUM_KERNEL


def _topk_merge(
    running_scores,
    running_indices,
    tile_scores,
    tile_indices,
    keep,
):
    """Merge a (rows, keep) running top-k with a (rows, *) tile top-k. Returns
    the new top-k. `tile_scores` / `tile_indices` may already be (rows, keep)
    (precomputed via `torch.topk` on the full tile-similarity row) or wider.
    """
    import cupy
    import torch

    combined_scores = cupy.concatenate([running_scores, tile_scores], axis=1)
    combined_indices = cupy.concatenate([running_indices, tile_indices], axis=1)
    combined_torch = torch.from_dlpack(combined_scores)
    merge_values, merge_pos = torch.topk(
        combined_torch, k=keep, dim=1, largest=True, sorted=False
    )
    new_scores = cupy.from_dlpack(merge_values)
    new_indices = cupy.take_along_axis(combined_indices, cupy.from_dlpack(merge_pos), axis=1)
    return new_scores, new_indices


def _tile_topk_dense(similarity_view, keep, candidate_offset_global, query_count, active_count):
    """Top-k inside one (Q, M) FP32 similarity tile -> global indices. Pads to
    `keep` columns when the tile is shorter than `keep`.
    """
    import cupy
    import torch

    if active_count >= keep:
        sim_torch = torch.from_dlpack(similarity_view)
        values, local = torch.topk(sim_torch, k=keep, dim=1, largest=True, sorted=False)
        return (
            cupy.from_dlpack(values),
            cupy.from_dlpack(local).astype(cupy.int32) + cupy.int32(candidate_offset_global),
        )
    pad = keep - active_count
    sub_global = cupy.arange(active_count, dtype=cupy.int32) + cupy.int32(
        candidate_offset_global
    )
    indices = cupy.concatenate(
        [
            cupy.broadcast_to(sub_global, (query_count, active_count)),
            cupy.full((query_count, pad), -1, dtype=cupy.int32),
        ],
        axis=1,
    )
    scores = cupy.concatenate(
        [similarity_view, cupy.full((query_count, pad), -cupy.inf, dtype=cupy.float32)],
        axis=1,
    )
    return scores, indices


def _drop_self(sorted_scores, sorted_indices, query_global_ids, num_neighbors):
    """Sort each row by descending score, then drop the (up-to-one) row whose
    index equals the query's own global id. Pure cupy, vectorized over rows.
    """
    import cupy

    is_self = sorted_indices == query_global_ids.reshape(-1, 1)
    has_self = cupy.any(is_self, axis=1, keepdims=True)
    self_pos = cupy.argmax(is_self.astype(cupy.int32), axis=1, keepdims=True)
    rows = sorted_scores.shape[0]
    output_columns = cupy.broadcast_to(
        cupy.arange(num_neighbors, dtype=cupy.int32), (rows, num_neighbors)
    )
    shift_mask = (output_columns >= self_pos) & has_self
    source_columns = output_columns + shift_mask.astype(cupy.int32)
    final_scores = cupy.take_along_axis(sorted_scores, source_columns, axis=1)
    final_indices = cupy.take_along_axis(sorted_indices, source_columns, axis=1)
    return final_indices, final_scores


def gt_stripe_dense(
    embeddings_host: np.ndarray,
    stripe_start: int,
    stripe_end: int,
    num_neighbors: int,
    query_tile_rows: int,
    candidate_tile_rows: int,
    log_prefix: str = "",
):
    """Compute top-k for `embeddings_host[stripe_start:stripe_end]` against
    the *whole* `embeddings_host` corpus, with double-buffered H2D streaming
    of candidate tiles. Returns `(indices, scores)` numpy arrays of shape
    `(stripe_end - stripe_start, num_neighbors)` in i32 / f32.

    Caller is responsible for picking the GPU (`CUDA_VISIBLE_DEVICES`).
    """
    import cupy
    import torch  # noqa: F401  (used inside helpers)

    total_vectors, dimensions = embeddings_host.shape
    stripe_size = stripe_end - stripe_start
    keep = num_neighbors + 1

    query_stripe = cupy.asarray(embeddings_host[stripe_start:stripe_end])

    pinned_holders = []
    pinned_views: list[np.ndarray] = []
    for _ in range(2):
        pinned = cupy.cuda.alloc_pinned_memory(candidate_tile_rows * dimensions * 2)
        pinned_holders.append(pinned)
        view = np.frombuffer(
            pinned, dtype=np.float16, count=candidate_tile_rows * dimensions
        ).reshape(candidate_tile_rows, dimensions)
        pinned_views.append(view)

    candidate_buffers = [
        cupy.empty((candidate_tile_rows, dimensions), dtype=cupy.float16),
        cupy.empty((candidate_tile_rows, dimensions), dtype=cupy.float16),
    ]
    similarity_buffer = cupy.empty(
        (query_tile_rows, candidate_tile_rows), dtype=cupy.float32
    )
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

    def stage(slot: int, tile_offset: int) -> int:
        count = min(candidate_tile_rows, total_vectors - tile_offset)
        copy_done[slot].synchronize()
        np.copyto(
            pinned_views[slot][:count],
            embeddings_host[tile_offset : tile_offset + count],
        )
        copy_stream.wait_event(compute_done[slot])
        candidate_buffers[slot][:count].set(pinned_views[slot][:count], stream=copy_stream)
        copy_done[slot].record(copy_stream)
        return count

    counts = [0, 0]
    for slot in range(min(2, len(candidate_offsets))):
        counts[slot] = stage(slot, candidate_offsets[slot])

    started = time.monotonic()
    for tile_idx, tile_offset in enumerate(candidate_offsets):
        slot = tile_idx % 2
        active_count = counts[slot]
        active_device = candidate_buffers[slot][:active_count]
        compute_stream.wait_event(copy_done[slot])

        with compute_stream:
            for query_start in range(0, stripe_size, query_tile_rows):
                query_end = min(query_start + query_tile_rows, stripe_size)
                query_count = query_end - query_start
                similarity_view = similarity_buffer[:query_count, :active_count]
                cupy.matmul(
                    query_stripe[query_start:query_end],
                    active_device.T,
                    out=similarity_view,
                )
                tile_scores, tile_indices = _tile_topk_dense(
                    similarity_view, keep, tile_offset, query_count, active_count
                )
                new_scores, new_indices = _topk_merge(
                    topk_scores[query_start:query_end],
                    topk_indices[query_start:query_end],
                    tile_scores,
                    tile_indices,
                    keep,
                )
                topk_scores[query_start:query_end] = new_scores
                topk_indices[query_start:query_end] = new_indices
            compute_done[slot].record(compute_stream)

        prefetch_idx = tile_idx + 2
        if prefetch_idx < len(candidate_offsets):
            counts[slot] = stage(slot, candidate_offsets[prefetch_idx])

        if (tile_idx + 1) % 32 == 0 or tile_idx + 1 == len(candidate_offsets):
            compute_stream.synchronize()
            cupy.get_default_memory_pool().free_all_blocks()
            elapsed = time.monotonic() - started
            done = (tile_idx + 1) * candidate_tile_rows
            rate = done / max(elapsed, 1e-3) / 1e6
            print(
                f"{log_prefix}tile {tile_idx + 1}/{len(candidate_offsets)} "
                f"elapsed {elapsed:.0f}s ({rate:.2f}M cand/s)",
                flush=True,
            )

    compute_stream.synchronize()

    sorted_order = cupy.argsort(-topk_scores, axis=1)
    sorted_scores = cupy.take_along_axis(topk_scores, sorted_order, axis=1)
    sorted_indices = cupy.take_along_axis(topk_indices, sorted_order, axis=1)
    query_global_ids = cupy.arange(stripe_start, stripe_end, dtype=cupy.int32)
    final_indices, final_scores = _drop_self(
        sorted_scores, sorted_indices, query_global_ids, num_neighbors
    )
    return cupy.asnumpy(final_indices), cupy.asnumpy(final_scores)


def gt_stripe_maxsim(
    token_bank_host: np.ndarray,
    section_offsets_host: np.ndarray,
    stripe_start_section: int,
    stripe_end_section: int,
    num_neighbors: int,
    query_tile_sections: int,
    candidate_tile_sections: int,
    log_prefix: str = "",
):
    """Compute MaxSim top-k for sections in
    `[stripe_start_section, stripe_end_section)` against the whole section
    corpus. Returns `(indices, scores)` numpy arrays of shape
    `(stripe_size, num_neighbors)`.

    `token_bank_host` is `(T_total, dim)` FP16; `section_offsets_host` is
    `(N+1,)` int32 with cumulative token counts (so section i's tokens live
    at rows `[offsets[i]:offsets[i+1]]`).
    """
    import cupy
    import torch  # noqa: F401

    total_sections = section_offsets_host.shape[0] - 1
    dimensions = token_bank_host.shape[1]
    stripe_size = stripe_end_section - stripe_start_section
    keep = num_neighbors + 1

    query_token_start = int(section_offsets_host[stripe_start_section])
    query_token_end = int(section_offsets_host[stripe_end_section])
    query_token_count = query_token_end - query_token_start
    query_tokens_device = cupy.asarray(token_bank_host[query_token_start:query_token_end])
    # Per-stripe local section offsets (zeroed against query_token_start).
    query_section_offsets_local = (
        section_offsets_host[stripe_start_section : stripe_end_section + 1]
        - query_token_start
    ).astype(np.int32)
    query_section_offsets_device = cupy.asarray(query_section_offsets_local)

    seg_max_kernel, seg_sum_kernel = _segment_kernels()

    candidate_starts = list(range(0, total_sections, candidate_tile_sections))

    # Pre-compute per-tile token byte budget so we can size pinned buffers.
    max_tile_tokens = 0
    for cand_start in candidate_starts:
        cand_end = min(cand_start + candidate_tile_sections, total_sections)
        tile_tokens = int(
            section_offsets_host[cand_end] - section_offsets_host[cand_start]
        )
        if tile_tokens > max_tile_tokens:
            max_tile_tokens = tile_tokens

    pinned_holders = []
    pinned_views: list[np.ndarray] = []
    for _ in range(2):
        pinned = cupy.cuda.alloc_pinned_memory(max_tile_tokens * dimensions * 2)
        pinned_holders.append(pinned)
        view = np.frombuffer(
            pinned, dtype=np.float16, count=max_tile_tokens * dimensions
        ).reshape(max_tile_tokens, dimensions)
        pinned_views.append(view)

    doc_token_buffers = [
        cupy.empty((max_tile_tokens, dimensions), dtype=cupy.float16),
        cupy.empty((max_tile_tokens, dimensions), dtype=cupy.float16),
    ]
    doc_offsets_buffers = [
        cupy.empty((candidate_tile_sections + 1,), dtype=cupy.int32),
        cupy.empty((candidate_tile_sections + 1,), dtype=cupy.int32),
    ]

    # Pre-allocated FP32 scratch buffers reused across tiles.
    sim_buffer = cupy.empty(
        (query_tile_sections * 64, max_tile_tokens), dtype=cupy.float32
    )
    # Bound on per-stripe query tokens per tile is `max query tokens` over the
    # query stripe. A safe upper bound for the buffer is the full stripe's
    # token count, but we tile queries by section count too — so size
    # generously to avoid mid-loop re-allocs.
    max_query_tokens_per_tile = int(
        max(
            section_offsets_host[
                min(stripe_start_section + query_tile_sections, stripe_end_section)
            ]
            - section_offsets_host[stripe_start_section],
            section_offsets_host[stripe_end_section]
            - section_offsets_host[
                max(stripe_end_section - query_tile_sections, stripe_start_section)
            ],
        )
    )
    # Re-size sim_buffer if it's too small for the worst-case (q_tokens, m_tokens).
    if sim_buffer.shape[0] < max_query_tokens_per_tile:
        sim_buffer = cupy.empty(
            (max_query_tokens_per_tile, max_tile_tokens), dtype=cupy.float32
        )
    per_token_max = cupy.empty(
        (max_query_tokens_per_tile, candidate_tile_sections), dtype=cupy.float32
    )
    score_out = cupy.empty(
        (query_tile_sections, candidate_tile_sections), dtype=cupy.float32
    )

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

    def stage(slot: int, cand_start: int):
        cand_end = min(cand_start + candidate_tile_sections, total_sections)
        section_count = cand_end - cand_start
        token_start = int(section_offsets_host[cand_start])
        token_end = int(section_offsets_host[cand_end])
        token_count = token_end - token_start
        copy_done[slot].synchronize()
        np.copyto(
            pinned_views[slot][:token_count],
            token_bank_host[token_start:token_end],
        )
        # Local doc offsets, zeroed against token_start.
        local_offsets = (
            section_offsets_host[cand_start : cand_end + 1] - token_start
        ).astype(np.int32)
        copy_stream.wait_event(compute_done[slot])
        doc_token_buffers[slot][:token_count].set(
            pinned_views[slot][:token_count], stream=copy_stream
        )
        doc_offsets_buffers[slot][: section_count + 1].set(
            local_offsets, stream=copy_stream
        )
        copy_done[slot].record(copy_stream)
        return section_count, token_count

    counts = [(0, 0), (0, 0)]
    for slot in range(min(2, len(candidate_starts))):
        counts[slot] = stage(slot, candidate_starts[slot])

    block = (16, 16, 1)
    started = time.monotonic()

    for tile_idx, cand_start in enumerate(candidate_starts):
        slot = tile_idx % 2
        section_count, token_count = counts[slot]
        if section_count == 0:
            continue
        compute_stream.wait_event(copy_done[slot])

        doc_tokens_dev = doc_token_buffers[slot][:token_count]
        doc_offsets_dev = doc_offsets_buffers[slot][: section_count + 1]

        with compute_stream:
            for q_section_start in range(0, stripe_size, query_tile_sections):
                q_section_end = min(q_section_start + query_tile_sections, stripe_size)
                q_count = q_section_end - q_section_start
                q_token_lo = int(query_section_offsets_local[q_section_start])
                q_token_hi = int(query_section_offsets_local[q_section_end])
                q_token_count = q_token_hi - q_token_lo
                if q_token_count == 0:
                    continue
                q_tokens_dev = query_tokens_device[q_token_lo:q_token_hi]
                # Local query offsets for this micro-batch (zeroed against q_token_lo).
                q_offsets_micro = (
                    query_section_offsets_device[q_section_start : q_section_end + 1]
                    - cupy.int32(q_token_lo)
                )

                sim_view = sim_buffer[:q_token_count, :token_count]
                cupy.matmul(q_tokens_dev, doc_tokens_dev.T, out=sim_view)

                pt_max_view = per_token_max[:q_token_count, :section_count]
                grid_max = (
                    (section_count + block[0] - 1) // block[0],
                    (q_token_count + block[1] - 1) // block[1],
                    1,
                )
                seg_max_kernel(
                    grid_max,
                    block,
                    (
                        sim_view,
                        doc_offsets_dev,
                        pt_max_view,
                        np.int32(q_token_count),
                        np.int32(section_count),
                        np.int32(sim_buffer.shape[1]),
                        np.int32(per_token_max.shape[1]),
                    ),
                )

                score_view = score_out[:q_count, :section_count]
                grid_sum = (
                    (section_count + block[0] - 1) // block[0],
                    (q_count + block[1] - 1) // block[1],
                    1,
                )
                seg_sum_kernel(
                    grid_sum,
                    block,
                    (
                        pt_max_view,
                        q_offsets_micro,
                        score_view,
                        np.int32(q_count),
                        np.int32(section_count),
                        np.int32(per_token_max.shape[1]),
                        np.int32(score_out.shape[1]),
                    ),
                )

                tile_scores, tile_indices = _tile_topk_dense(
                    score_view, keep, cand_start, q_count, section_count
                )
                new_scores, new_indices = _topk_merge(
                    topk_scores[q_section_start:q_section_end],
                    topk_indices[q_section_start:q_section_end],
                    tile_scores,
                    tile_indices,
                    keep,
                )
                topk_scores[q_section_start:q_section_end] = new_scores
                topk_indices[q_section_start:q_section_end] = new_indices
            compute_done[slot].record(compute_stream)

        prefetch_idx = tile_idx + 2
        if prefetch_idx < len(candidate_starts):
            counts[slot] = stage(slot, candidate_starts[prefetch_idx])

        if (tile_idx + 1) % 32 == 0 or tile_idx + 1 == len(candidate_starts):
            compute_stream.synchronize()
            cupy.get_default_memory_pool().free_all_blocks()
            elapsed = time.monotonic() - started
            done = (tile_idx + 1) * candidate_tile_sections
            rate = done / max(elapsed, 1e-3) / 1e6
            print(
                f"{log_prefix}tile {tile_idx + 1}/{len(candidate_starts)} "
                f"elapsed {elapsed:.0f}s ({rate:.2f}M sect/s)",
                flush=True,
            )

    compute_stream.synchronize()

    sorted_order = cupy.argsort(-topk_scores, axis=1)
    sorted_scores = cupy.take_along_axis(topk_scores, sorted_order, axis=1)
    sorted_indices = cupy.take_along_axis(topk_indices, sorted_order, axis=1)
    query_global_ids = cupy.arange(
        stripe_start_section, stripe_end_section, dtype=cupy.int32
    )
    final_indices, final_scores = _drop_self(
        sorted_scores, sorted_indices, query_global_ids, num_neighbors
    )
    return cupy.asnumpy(final_indices), cupy.asnumpy(final_scores)


# ---------------------------------------------------------------------------
# Consumer-facing classes: load corpus once, search many.
# ---------------------------------------------------------------------------


def _read_header(path: Path) -> tuple[int, int]:
    blob = resolve_lfs_pointer(path)
    with open(blob, "rb") as file:
        rows, columns = struct.unpack("<II", file.read(8))
    return rows, columns


def _load_dense_corpus_to_host(
    model_root: Path, suffix: str, shards: list[CollectionShard]
) -> tuple[np.ndarray, int]:
    """Same data-load contract as `ground_truth.load_collection` but kept here
    so `DenseRetriever` doesn't need to import the GT script.
    """
    if not shards:
        raise ValueError(f"no shards under {model_root}")
    _, dimensions = _read_header(shards[0].path)
    total = sum(s.row_count for s in shards)
    embeddings = np.empty((total, dimensions), dtype=np.float16)
    for shard in shards:
        blob = resolve_lfs_pointer(shard.path)
        with open(blob, "rb") as file:
            file.seek(8)
            destination = embeddings[shard.row_offset : shard.row_offset + shard.row_count]
            file.readinto(memoryview(destination))  # type: ignore[arg-type]
    bad = ~np.isfinite(embeddings).all(axis=1)
    if bad.any():
        embeddings[bad] = 0
    return embeddings, dimensions


def _load_maxsim_corpus_to_host(
    model_root: Path, suffix: str
) -> tuple[np.ndarray, np.ndarray, list[CollectionShard], int]:
    """Walk `*.{suffix}.sections.f16bin` + `*.sections.offsets.ibin` shards in
    canonical order. Return `(token_bank, section_offsets, shards, dimensions)`.

    `section_offsets` has shape `(total_sections + 1,)` int32 cumulative across
    all shards (so section i's tokens are at `token_bank[offsets[i]:offsets[i+1]]`).
    Each `CollectionShard.row_count` is the number of *sections* in that shard,
    `row_offset` is the cumulative section count.
    """
    if not model_root.is_dir():
        raise FileNotFoundError(f"no model directory at {model_root}")
    shards: list[CollectionShard] = []
    cumulative_sections = 0
    cumulative_tokens = 0
    section_offsets_chunks: list[np.ndarray] = []
    section_offsets_chunks.append(np.zeros(1, dtype=np.int32))
    token_chunks: list[tuple[int, int, Path]] = []  # (offset_in_bank, token_count, path)
    dimensions: int | None = None
    for wiki_dir in sorted(model_root.iterdir()):
        if not wiki_dir.is_dir():
            continue
        for path in sorted(wiki_dir.glob(f"*.{suffix}.sections.f16bin")):
            stem = path.name[: -len(f".{suffix}.sections.f16bin")]
            tokens, dim = _read_header(path)
            if dimensions is None:
                dimensions = dim
            elif dim != dimensions:
                raise ValueError(f"{path}: dim {dim} != expected {dimensions}")
            offsets_path = wiki_dir / f"{stem}.{suffix}.sections.offsets.ibin"
            if not offsets_path.is_file():
                raise FileNotFoundError(f"missing offsets file: {offsets_path}")
            offsets_blob = resolve_lfs_pointer(offsets_path)
            with open(offsets_blob, "rb") as file:
                rows, _cols = struct.unpack("<II", file.read(8))
                local_offsets = np.frombuffer(
                    file.read(), dtype=np.int32, count=rows
                ).copy()
            n_sections = rows - 1
            shifted = (local_offsets + cumulative_tokens).astype(np.int32)
            # `local_offsets` already starts at 0; we drop the first element of
            # subsequent chunks since `cumulative_tokens` provides the seam.
            section_offsets_chunks.append(shifted[1:])
            shards.append(
                CollectionShard(
                    wikiname=wiki_dir.name,
                    stem=stem,
                    path=path,
                    row_offset=cumulative_sections,
                    row_count=n_sections,
                )
            )
            token_chunks.append((cumulative_tokens, tokens, path))
            cumulative_sections += n_sections
            cumulative_tokens += tokens
    if dimensions is None:
        raise FileNotFoundError(f"no `.{suffix}.sections.f16bin` files under {model_root}")

    token_bank = np.empty((cumulative_tokens, dimensions), dtype=np.float16)
    for token_offset, token_count, path in token_chunks:
        blob = resolve_lfs_pointer(path)
        with open(blob, "rb") as file:
            file.seek(8)
            destination = token_bank[token_offset : token_offset + token_count]
            file.readinto(memoryview(destination))  # type: ignore[arg-type]
    bad = ~np.isfinite(token_bank).all(axis=1)
    if bad.any():
        token_bank[bad] = 0
    section_offsets = np.concatenate(section_offsets_chunks).astype(np.int32)
    return token_bank, section_offsets, shards, dimensions


class DenseRetriever:
    """Brute-force exact cosine top-k for a dense embedding collection.

    Loads the entire `(N, dim)` FP16 corpus to one GPU at construction. Each
    `search()` call runs a single matmul + top-k against that resident corpus.

    For 60M × 1024 FP16 (~120 GB) the corpus does NOT fit on a single 80 GB
    H100 — quantize on-disk before instantiating, or use the multi-GPU
    `gt_stripe_dense` path directly. The single-resident-corpus class is
    designed for moderate-size collections (≤ tens of GB).
    """

    def __init__(
        self,
        model_root: str | Path,
        suffix: str = "body",
        device_id: int = 0,
    ):
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        import cupy

        model_root = Path(model_root)
        self.model_root = model_root
        self.suffix = suffix
        self.shards = discover_collection(model_root, suffix)
        embeddings_host, self.dimensions = _load_dense_corpus_to_host(
            model_root, suffix, self.shards
        )
        self.total_vectors = embeddings_host.shape[0]
        # Resident on GPU — caller pays the upfront cost; subsequent searches
        # are bandwidth-bound on the matmul alone.
        self.corpus_device = cupy.asarray(embeddings_host)

    def search(
        self, query_vectors: np.ndarray, k: int = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        """`query_vectors`: `(Q, dim)` FP16, already L2-normalized.
        Returns `(scores, indices)` — both `(Q, k)` numpy arrays.
        """
        import cupy
        import torch  # noqa: F401

        if query_vectors.ndim != 2 or query_vectors.shape[1] != self.dimensions:
            raise ValueError(
                f"queries shape {query_vectors.shape} != (?, {self.dimensions})"
            )
        queries_dev = cupy.asarray(query_vectors.astype(np.float16, copy=False))
        sim = cupy.matmul(queries_dev, self.corpus_device.T, dtype=cupy.float32)
        sim_torch = torch.from_dlpack(sim)
        values, local = torch.topk(sim_torch, k=k, dim=1, largest=True, sorted=True)
        return cupy.asnumpy(cupy.from_dlpack(values)), cupy.asnumpy(
            cupy.from_dlpack(local).astype(cupy.int32)
        )


class MaxSimRetriever:
    """Brute-force exact MaxSim top-k for a multi-vector section corpus.

    Loads the full token bank + section offsets to one GPU at construction.
    `search()` accepts a query in the same `(token_bank, offsets)` shape and
    runs matmul + segment-max + segment-sum + top-k against the resident
    corpus.

    For 60 M sections × 128 dim with average 3.4 tokens/section the resident
    bank is ~52 GB FP16 — fits on H100 with margin only after FP8/int8
    quantization, or on B200 / multi-GPU for raw FP16.
    """

    def __init__(
        self,
        model_root: str | Path,
        suffix: str = "body",
        device_id: int = 0,
    ):
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        import cupy

        model_root = Path(model_root)
        self.model_root = model_root
        self.suffix = suffix
        (
            token_bank_host,
            section_offsets_host,
            self.shards,
            self.dimensions,
        ) = _load_maxsim_corpus_to_host(model_root, suffix)
        self.total_sections = section_offsets_host.shape[0] - 1
        self.total_tokens = token_bank_host.shape[0]
        self.token_bank_device = cupy.asarray(token_bank_host)
        self.section_offsets_device = cupy.asarray(section_offsets_host)

    def search(
        self,
        query_token_bank: np.ndarray,
        query_section_offsets: np.ndarray,
        k: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """`query_token_bank`: `(T_q, dim)` FP16. `query_section_offsets`:
        `(Q+1,)` int32 cumulative. Returns `(scores, indices)` — `(Q, k)`.
        """
        import cupy
        import torch  # noqa: F401

        if query_token_bank.ndim != 2 or query_token_bank.shape[1] != self.dimensions:
            raise ValueError(
                f"queries shape {query_token_bank.shape} != (?, {self.dimensions})"
            )
        n_queries = query_section_offsets.shape[0] - 1
        seg_max_kernel, seg_sum_kernel = _segment_kernels()

        queries_dev = cupy.asarray(query_token_bank.astype(np.float16, copy=False))
        q_offsets_dev = cupy.asarray(query_section_offsets.astype(np.int32, copy=False))

        sim = cupy.matmul(queries_dev, self.token_bank_device.T, dtype=cupy.float32)
        per_token_max = cupy.empty(
            (queries_dev.shape[0], self.total_sections), dtype=cupy.float32
        )
        block = (16, 16, 1)
        grid_max = (
            (self.total_sections + block[0] - 1) // block[0],
            (queries_dev.shape[0] + block[1] - 1) // block[1],
            1,
        )
        seg_max_kernel(
            grid_max,
            block,
            (
                sim,
                self.section_offsets_device,
                per_token_max,
                np.int32(queries_dev.shape[0]),
                np.int32(self.total_sections),
                np.int32(sim.shape[1]),
                np.int32(per_token_max.shape[1]),
            ),
        )

        scores = cupy.empty((n_queries, self.total_sections), dtype=cupy.float32)
        grid_sum = (
            (self.total_sections + block[0] - 1) // block[0],
            (n_queries + block[1] - 1) // block[1],
            1,
        )
        seg_sum_kernel(
            grid_sum,
            block,
            (
                per_token_max,
                q_offsets_dev,
                scores,
                np.int32(n_queries),
                np.int32(self.total_sections),
                np.int32(per_token_max.shape[1]),
                np.int32(scores.shape[1]),
            ),
        )

        scores_torch = torch.from_dlpack(scores)
        values, local = torch.topk(scores_torch, k=k, dim=1, largest=True, sorted=True)
        return cupy.asnumpy(cupy.from_dlpack(values)), cupy.asnumpy(
            cupy.from_dlpack(local).astype(cupy.int32)
        )
