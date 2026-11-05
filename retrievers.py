"""Brute-force retrievers and shared GPU kernels.

Two retrieval families share one streaming-top-k template — different inner
score kernel, same outer machinery (resident query stripe, double-buffered
H2D candidate tiles, pre-allocated FP32 similarity buffer):

  - **Dense** — cosine top-k over an `(N, dim)` FP16 corpus. Inner kernel is
    one FP16 matmul into a pre-allocated FP32 ``out=`` buffer.
  - **MaxSim** — ColBERT-style late interaction over a multi-vector corpus
    (``(T_total, dim)`` token bank + ``(N+1,)`` section offsets). Inner
    kernel is matmul + segment-max over ragged doc-token offsets +
    segment-sum over ragged query-token offsets, via two small CUDA
    RawKernels.

Each path exposes itself two ways:
  - ``gt_stripe_*`` — one-shot per-GPU stripe worker used by
    ``ground_truth.py`` for the all-vs-all corpus sweep.
  - ``DenseRetriever`` / ``MaxSimRetriever`` — consumer-facing search,
    corpus loaded once at construction and held resident on a single GPU.
    ``search()`` runs a single matmul + top-k against the resident corpus.
"""

from __future__ import annotations

import struct
import time
from pathlib import Path

import numpy as np

from usearchwiki import (
    CollectionShard,
    discover_collection,
)


# region: Common utilities


def _read_header(path: Path) -> tuple[int, int]:
    blob = path
    with open(blob, "rb") as file:
        rows, columns = struct.unpack("<II", file.read(8))
    return rows, columns


# endregion


# region: GPU kernels

# Two ragged-segment reductions for MaxSim. Each thread handles one
# (row, segment) cell; the inner loop is bounded by the segment width
# (median 3, p99 ~16 for FineWiki sections, so a single warp easily covers
# the worst rows without divergence pain).

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
    // 64-bit offset arithmetic: with a 7.7M-section stripe and
    // query_tile_sections=2048, the max sliding-window query-token count
    // can hit ~32K; combined with max_tile_tokens ~100K, row * row_stride
    // exceeds INT32_MAX. The output stride is bounded by
    // candidate_tile_sections (≤ ~32K), so 32-bit math is fine for `out`.
    const float* row_ptr = values + (long long)row * (long long)row_stride;
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
    int n_segments, int n_cols, int row_stride, int out_stride,
    int offset_base
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int seg = blockIdx.y * blockDim.y + threadIdx.y;
    if (seg >= n_segments || col >= n_cols) return;
    int start = offsets[seg]     - offset_base;
    int end   = offsets[seg + 1] - offset_base;
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
    """Compile and cache the two RawKernels on first use. Lazy because cupy
    initializes a CUDA context on import — we want that deferred until inside
    the worker process (post-fork).
    """
    global _SEGMENT_MAX_KERNEL, _SEGMENT_SUM_KERNEL
    if _SEGMENT_MAX_KERNEL is None:
        import cupy

        _SEGMENT_MAX_KERNEL = cupy.RawKernel(_SEGMENT_MAX_SRC, "segment_max_2d")
        _SEGMENT_SUM_KERNEL = cupy.RawKernel(_SEGMENT_SUM_SRC, "segment_sum_2d")
    return _SEGMENT_MAX_KERNEL, _SEGMENT_SUM_KERNEL


# endregion


# region: Top-k primitives


def _drop_self_match(sorted_scores, sorted_indices, query_global_ids, num_neighbors):
    """Drop the (up-to-one) row whose index equals the query's own global
    id — call after `cupy.argsort(-scores)`. Pure cupy, vectorized over rows.
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


def _topk_merge(running_scores, running_indices, tile_scores, tile_indices, keep):
    """Merge a `(rows, keep)` running top-k with a `(rows, *)` tile top-k.
    Returns the new `(rows, keep)` top-k. Allocates new arrays — used by the
    dense path, where matmul size makes the per-iter alloc cost negligible.
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
    new_indices = cupy.take_along_axis(
        combined_indices, cupy.from_dlpack(merge_pos), axis=1
    )
    return new_scores, new_indices


def _tile_topk(
    similarity_view, keep, candidate_offset_global, query_count, active_count
):
    """Top-k inside one `(Q, M)` FP32 similarity tile, lifting local indices
    to global. Pads to `keep` columns when the tile is shorter than `keep`.
    """
    import cupy
    import torch

    if active_count >= keep:
        sim_torch = torch.from_dlpack(similarity_view)
        values, local = torch.topk(
            sim_torch, k=keep, dim=1, largest=True, sorted=False
        )
        return (
            cupy.from_dlpack(values),
            cupy.from_dlpack(local).astype(cupy.int32)
            + cupy.int32(candidate_offset_global),
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
        [
            similarity_view,
            cupy.full((query_count, pad), -cupy.inf, dtype=cupy.float32),
        ],
        axis=1,
    )
    return scores, indices


# endregion


# region: Dense path


class _DenseStripeRunner:
    """Per-GPU runner for the dense all-vs-all top-k sweep.

    Owns the resident query stripe, the double-buffered H2D candidate tiles,
    the pre-allocated FP32 similarity buffer, and the running top-k state.
    One instance per ``gt_stripe_dense()`` call.
    """

    def __init__(
        self,
        embeddings_host,
        stripe_start,
        stripe_end,
        num_neighbors,
        query_tile_rows,
        candidate_tile_rows,
        log_prefix,
    ):
        import cupy

        self.embeddings_host = embeddings_host
        self.stripe_start = stripe_start
        self.stripe_end = stripe_end
        self.stripe_size = stripe_end - stripe_start
        self.num_neighbors = num_neighbors
        self.query_tile_rows = query_tile_rows
        self.candidate_tile_rows = candidate_tile_rows
        self.log_prefix = log_prefix
        self.keep = num_neighbors + 1

        total_vectors, dimensions = embeddings_host.shape
        self.total_vectors = total_vectors
        self.dimensions = dimensions

        self.query_stripe = cupy.asarray(embeddings_host[stripe_start:stripe_end])

        self.pinned_holders: list = []
        self.pinned_views: list[np.ndarray] = []
        for _ in range(2):
            pinned = cupy.cuda.alloc_pinned_memory(
                candidate_tile_rows * dimensions * 2
            )
            self.pinned_holders.append(pinned)
            view = np.frombuffer(
                pinned, dtype=np.float16, count=candidate_tile_rows * dimensions
            ).reshape(candidate_tile_rows, dimensions)
            self.pinned_views.append(view)
        self.candidate_buffers = [
            cupy.empty((candidate_tile_rows, dimensions), dtype=cupy.float16),
            cupy.empty((candidate_tile_rows, dimensions), dtype=cupy.float16),
        ]
        self.similarity_buffer = cupy.empty(
            (query_tile_rows, candidate_tile_rows), dtype=cupy.float32
        )

        self.topk_scores = cupy.full(
            (self.stripe_size, self.keep), -cupy.inf, dtype=cupy.float32
        )
        self.topk_indices = cupy.full(
            (self.stripe_size, self.keep), -1, dtype=cupy.int32
        )

        self.copy_stream = cupy.cuda.Stream(non_blocking=True)
        self.compute_stream = cupy.cuda.Stream(non_blocking=True)
        self.copy_done = [cupy.cuda.Event(disable_timing=True) for _ in range(2)]
        self.compute_done = [cupy.cuda.Event(disable_timing=True) for _ in range(2)]

        self.candidate_offsets = list(range(0, total_vectors, candidate_tile_rows))

    def _stage_tile(self, slot: int, tile_offset: int) -> int:
        count = min(self.candidate_tile_rows, self.total_vectors - tile_offset)
        self.copy_done[slot].synchronize()
        np.copyto(
            self.pinned_views[slot][:count],
            self.embeddings_host[tile_offset : tile_offset + count],
        )
        self.copy_stream.wait_event(self.compute_done[slot])
        self.candidate_buffers[slot][:count].set(
            self.pinned_views[slot][:count], stream=self.copy_stream
        )
        self.copy_done[slot].record(self.copy_stream)
        return count

    def _score_microbatch(
        self, active_device, active_count, tile_offset, query_start, query_end
    ):
        import cupy

        query_count = query_end - query_start
        similarity_view = self.similarity_buffer[:query_count, :active_count]
        cupy.matmul(
            self.query_stripe[query_start:query_end],
            active_device.T,
            out=similarity_view,
        )
        tile_scores, tile_indices = _tile_topk(
            similarity_view, self.keep, tile_offset, query_count, active_count
        )
        new_scores, new_indices = _topk_merge(
            self.topk_scores[query_start:query_end],
            self.topk_indices[query_start:query_end],
            tile_scores,
            tile_indices,
            self.keep,
        )
        self.topk_scores[query_start:query_end] = new_scores
        self.topk_indices[query_start:query_end] = new_indices

    def _maybe_log_progress(self, tile_idx: int, started: float):
        import cupy

        last = tile_idx + 1 == len(self.candidate_offsets)
        if (tile_idx + 1) % 32 != 0 and not last:
            return
        self.compute_stream.synchronize()
        cupy.get_default_memory_pool().free_all_blocks()
        elapsed = time.monotonic() - started
        done = (tile_idx + 1) * self.candidate_tile_rows
        rate = done / max(elapsed, 1e-3) / 1e6
        print(
            f"{self.log_prefix}tile {tile_idx + 1}/{len(self.candidate_offsets)} "
            f"elapsed {elapsed:.0f}s ({rate:.2f}M cand/s)",
            flush=True,
        )

    def _finalize(self) -> tuple[np.ndarray, np.ndarray]:
        import cupy

        sorted_order = cupy.argsort(-self.topk_scores, axis=1)
        sorted_scores = cupy.take_along_axis(self.topk_scores, sorted_order, axis=1)
        sorted_indices = cupy.take_along_axis(self.topk_indices, sorted_order, axis=1)
        query_global_ids = cupy.arange(
            self.stripe_start, self.stripe_end, dtype=cupy.int32
        )
        final_indices, final_scores = _drop_self_match(
            sorted_scores, sorted_indices, query_global_ids, self.num_neighbors
        )
        return cupy.asnumpy(final_indices), cupy.asnumpy(final_scores)

    def run(self) -> tuple[np.ndarray, np.ndarray]:
        counts = [0, 0]
        for slot in range(min(2, len(self.candidate_offsets))):
            counts[slot] = self._stage_tile(slot, self.candidate_offsets[slot])

        started = time.monotonic()
        for tile_idx, tile_offset in enumerate(self.candidate_offsets):
            slot = tile_idx % 2
            active_count = counts[slot]
            active_device = self.candidate_buffers[slot][:active_count]
            self.compute_stream.wait_event(self.copy_done[slot])

            with self.compute_stream:
                for query_start in range(0, self.stripe_size, self.query_tile_rows):
                    query_end = min(
                        query_start + self.query_tile_rows, self.stripe_size
                    )
                    self._score_microbatch(
                        active_device, active_count, tile_offset, query_start, query_end
                    )
                self.compute_done[slot].record(self.compute_stream)

            prefetch_idx = tile_idx + 2
            if prefetch_idx < len(self.candidate_offsets):
                counts[slot] = self._stage_tile(
                    slot, self.candidate_offsets[prefetch_idx]
                )

            self._maybe_log_progress(tile_idx, started)

        self.compute_stream.synchronize()
        return self._finalize()


def gt_stripe_dense(
    embeddings_host: np.ndarray,
    stripe_start: int,
    stripe_end: int,
    num_neighbors: int,
    query_tile_rows: int,
    candidate_tile_rows: int,
    log_prefix: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact top-k for ``embeddings_host[stripe_start:stripe_end]``
    against the *whole* ``embeddings_host`` corpus, with double-buffered H2D
    streaming of candidate tiles. Returns ``(indices, scores)`` numpy arrays
    of shape ``(stripe_end - stripe_start, num_neighbors)`` in i32 / f32.

    Caller picks the GPU via ``CUDA_VISIBLE_DEVICES``.
    """
    return _DenseStripeRunner(
        embeddings_host,
        stripe_start,
        stripe_end,
        num_neighbors,
        query_tile_rows,
        candidate_tile_rows,
        log_prefix,
    ).run()


def load_dense_corpus(
    model_root: Path, suffix: str, shards: list[CollectionShard]
) -> tuple[np.ndarray, int]:
    """Load every shard of a dense collection into one host FP16 array.
    Sanitizes non-finite rows (a handful of WikiVerse f16bin files contain
    stray NaN/Inf — see project memory).
    """
    if not shards:
        raise ValueError(f"no shards under {model_root}")
    _, dimensions = _read_header(shards[0].path)
    total = sum(shard.row_count for shard in shards)
    embeddings = np.empty((total, dimensions), dtype=np.float16)
    for shard in shards:
        blob = shard.path
        with open(blob, "rb") as file:
            file.seek(8)
            destination = embeddings[
                shard.row_offset : shard.row_offset + shard.row_count
            ]
            file.readinto(memoryview(destination))  # type: ignore[arg-type]
    bad = ~np.isfinite(embeddings).all(axis=1)
    if bad.any():
        embeddings[bad] = 0
    return embeddings, dimensions


class DenseRetriever:
    """Brute-force exact cosine top-k for a dense embedding collection.

    Loads the entire ``(N, dim)`` FP16 corpus to one GPU at construction.
    Each ``search()`` call runs a single matmul + top-k against the resident
    corpus. For ~120 GB collections (60M × 1024 FP16) the corpus does NOT
    fit on a single 80 GB H100 — quantize on-disk before instantiating, or
    use the multi-GPU ``gt_stripe_dense`` path directly. This class is
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
        embeddings_host, self.dimensions = load_dense_corpus(
            model_root, suffix, self.shards
        )
        self.total_vectors = embeddings_host.shape[0]
        # Resident on GPU — subsequent searches are bound by the matmul.
        self.corpus_device = cupy.asarray(embeddings_host)

    def search(
        self, query_vectors: np.ndarray, k: int = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        """``query_vectors``: ``(Q, dim)`` FP16, already L2-normalized.
        Returns ``(scores, indices)`` — both ``(Q, k)`` numpy arrays.
        """
        import cupy
        import torch

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


# endregion


# region: MaxSim path


class _MaxSimStripeRunner:
    """Per-GPU runner for the MaxSim all-vs-all top-k sweep.

    Owns the resident query-stripe token bank, the double-buffered H2D
    candidate token tiles, the pre-allocated FP32 similarity / per-token-max
    / score buffers, and the running top-k state. One instance per
    ``gt_stripe_maxsim()`` call.

    The hot loop is bound by Python + allocator overhead at ~1 ms/iter, so
    every per-microbatch buffer is pre-allocated once in `__init__` and
    reused. All torch ops in `_score_microbatch` run on the cupy compute
    stream (via ``torch.cuda.ExternalStream``) — without that binding,
    torch ops would queue on torch's default stream and race with cupy
    writes through the merge buffer.
    """

    _BLOCK = (16, 16, 1)

    def __init__(
        self,
        token_bank_host,
        section_offsets_host,
        stripe_start_section,
        stripe_end_section,
        num_neighbors,
        query_tile_sections,
        candidate_tile_sections,
        log_prefix,
    ):
        import cupy
        import torch

        self.token_bank_host = token_bank_host
        self.section_offsets_host = section_offsets_host
        self.stripe_start_section = stripe_start_section
        self.stripe_end_section = stripe_end_section
        self.stripe_size = stripe_end_section - stripe_start_section
        self.num_neighbors = num_neighbors
        self.query_tile_sections = query_tile_sections
        self.candidate_tile_sections = candidate_tile_sections
        self.log_prefix = log_prefix
        self.keep = num_neighbors + 1

        self.total_sections = section_offsets_host.shape[0] - 1
        self.dimensions = token_bank_host.shape[1]

        # Resident query-stripe token bank on GPU.
        query_token_start = int(section_offsets_host[stripe_start_section])
        query_token_end = int(section_offsets_host[stripe_end_section])
        self.query_tokens_device = cupy.asarray(
            token_bank_host[query_token_start:query_token_end]
        )
        # Per-stripe local section offsets, zeroed against query_token_start.
        self.query_section_offsets_local = (
            section_offsets_host[stripe_start_section : stripe_end_section + 1]
            - query_token_start
        ).astype(np.int32)
        self.query_section_offsets_device = cupy.asarray(
            self.query_section_offsets_local
        )

        self.segment_max_kernel, self.segment_sum_kernel = _segment_kernels()

        self.candidate_starts = list(
            range(0, self.total_sections, candidate_tile_sections)
        )
        max_tile_tokens = self._scan_max_tile_tokens()
        max_query_tokens_per_tile = self._scan_max_query_tokens()

        # Pinned host scratch + resident device buffers for candidate tiles.
        self.pinned_holders: list = []
        self.pinned_views: list[np.ndarray] = []
        for _ in range(2):
            pinned = cupy.cuda.alloc_pinned_memory(
                max_tile_tokens * self.dimensions * 2
            )
            self.pinned_holders.append(pinned)
            view = np.frombuffer(
                pinned, dtype=np.float16, count=max_tile_tokens * self.dimensions
            ).reshape(max_tile_tokens, self.dimensions)
            self.pinned_views.append(view)
        self.doc_token_buffers = [
            cupy.empty((max_tile_tokens, self.dimensions), dtype=cupy.float16),
            cupy.empty((max_tile_tokens, self.dimensions), dtype=cupy.float16),
        ]
        self.doc_offsets_buffers = [
            cupy.empty((candidate_tile_sections + 1,), dtype=cupy.int32),
            cupy.empty((candidate_tile_sections + 1,), dtype=cupy.int32),
        ]

        # Pre-allocated FP32 scratch for the inner loop.
        self.sim_buffer = cupy.empty(
            (max_query_tokens_per_tile, max_tile_tokens), dtype=cupy.float32
        )
        self.per_token_max = cupy.empty(
            (max_query_tokens_per_tile, candidate_tile_sections), dtype=cupy.float32
        )
        self.score_out = cupy.empty(
            (query_tile_sections, candidate_tile_sections), dtype=cupy.float32
        )

        self.topk_scores = cupy.full(
            (self.stripe_size, self.keep), -cupy.inf, dtype=cupy.float32
        )
        self.topk_indices = cupy.full(
            (self.stripe_size, self.keep), -1, dtype=cupy.int32
        )

        # Pre-allocated merge scratch — eliminates allocator churn in the
        # hot loop. (Q, 2*keep) is small but called per microbatch.
        self.combined_scores = cupy.empty(
            (query_tile_sections, 2 * self.keep), dtype=cupy.float32
        )
        self.combined_indices = cupy.empty(
            (query_tile_sections, 2 * self.keep), dtype=cupy.int32
        )
        self.tile_values_t = torch.empty(
            (query_tile_sections, self.keep), dtype=torch.float32, device="cuda"
        )
        self.tile_local_t = torch.empty(
            (query_tile_sections, self.keep), dtype=torch.int64, device="cuda"
        )
        self.merge_pos_t = torch.empty(
            (query_tile_sections, self.keep), dtype=torch.int64, device="cuda"
        )
        # DLPack-wrap pre-allocated cupy buffers as torch views *once*:
        # re-wrapping per-iter trips torch's DLPack-consume guard on some
        # versions and surfaces as cudaErrorIllegalAddress later.
        self.combined_scores_t = torch.from_dlpack(self.combined_scores)
        self.combined_indices_t = torch.from_dlpack(self.combined_indices)
        self.topk_scores_t = torch.from_dlpack(self.topk_scores)
        self.topk_indices_t = torch.from_dlpack(self.topk_indices)

        self.copy_stream = cupy.cuda.Stream(non_blocking=True)
        self.compute_stream = cupy.cuda.Stream(non_blocking=True)
        self.copy_done = [cupy.cuda.Event(disable_timing=True) for _ in range(2)]
        self.compute_done = [cupy.cuda.Event(disable_timing=True) for _ in range(2)]
        self.torch_compute_stream = torch.cuda.ExternalStream(self.compute_stream.ptr)

    def _scan_max_tile_tokens(self) -> int:
        max_tile_tokens = 0
        for candidate_start in self.candidate_starts:
            candidate_end = min(
                candidate_start + self.candidate_tile_sections, self.total_sections
            )
            tile_tokens = int(
                self.section_offsets_host[candidate_end]
                - self.section_offsets_host[candidate_start]
            )
            if tile_tokens > max_tile_tokens:
                max_tile_tokens = tile_tokens
        return max_tile_tokens

    def _scan_max_query_tokens(self) -> int:
        # True upper bound on query-token count for any sliding window of
        # `query_tile_sections` sections within the stripe. The corpus has
        # outlier sections (max ~1000 tokens) so a coarse heuristic is unsafe.
        if self.stripe_size <= self.query_tile_sections:
            return int(
                self.section_offsets_host[self.stripe_end_section]
                - self.section_offsets_host[self.stripe_start_section]
            )
        starts = np.arange(
            self.stripe_start_section,
            self.stripe_end_section - self.query_tile_sections + 1,
        )
        ends = starts + self.query_tile_sections
        window_tokens = (
            self.section_offsets_host[ends] - self.section_offsets_host[starts]
        )
        return int(window_tokens.max())

    def _stage_tile(self, slot: int, candidate_start: int) -> tuple[int, int]:
        candidate_end = min(
            candidate_start + self.candidate_tile_sections, self.total_sections
        )
        section_count = candidate_end - candidate_start
        token_start = int(self.section_offsets_host[candidate_start])
        token_end = int(self.section_offsets_host[candidate_end])
        token_count = token_end - token_start
        self.copy_done[slot].synchronize()
        np.copyto(
            self.pinned_views[slot][:token_count],
            self.token_bank_host[token_start:token_end],
        )
        local_offsets = (
            self.section_offsets_host[candidate_start : candidate_end + 1]
            - token_start
        ).astype(np.int32)
        self.copy_stream.wait_event(self.compute_done[slot])
        self.doc_token_buffers[slot][:token_count].set(
            self.pinned_views[slot][:token_count], stream=self.copy_stream
        )
        self.doc_offsets_buffers[slot][: section_count + 1].set(
            local_offsets, stream=self.copy_stream
        )
        self.copy_done[slot].record(self.copy_stream)
        return section_count, token_count

    def _score_microbatch(
        self,
        slot: int,
        section_count: int,
        token_count: int,
        candidate_start_i32,
        query_section_start: int,
        query_section_end: int,
    ):
        import cupy
        import torch

        query_section_count = query_section_end - query_section_start
        query_token_start = int(
            self.query_section_offsets_local[query_section_start]
        )
        query_token_end = int(self.query_section_offsets_local[query_section_end])
        query_token_count = query_token_end - query_token_start
        if query_token_count == 0:
            return

        doc_tokens_dev = self.doc_token_buffers[slot][:token_count]
        doc_offsets_dev = self.doc_offsets_buffers[slot][: section_count + 1]
        query_tokens_dev = self.query_tokens_device[query_token_start:query_token_end]

        # Step 1 — matmul into the FP32 sim buffer.
        sim_view = self.sim_buffer[:query_token_count, :token_count]
        cupy.matmul(query_tokens_dev, doc_tokens_dev.T, out=sim_view)

        # Step 2 — per-(q_token, doc) max over the doc-token segment.
        per_token_max_view = self.per_token_max[:query_token_count, :section_count]
        block = self._BLOCK
        grid_max = (
            (section_count + block[0] - 1) // block[0],
            (query_token_count + block[1] - 1) // block[1],
            1,
        )
        self.segment_max_kernel(
            grid_max,
            block,
            (
                sim_view,
                doc_offsets_dev,
                per_token_max_view,
                np.int32(query_token_count),
                np.int32(section_count),
                np.int32(self.sim_buffer.shape[1]),
                np.int32(self.per_token_max.shape[1]),
            ),
        )

        # Step 3 — sum over query tokens within each query section.
        score_view = self.score_out[:query_section_count, :section_count]
        grid_sum = (
            (section_count + block[0] - 1) // block[0],
            (query_section_count + block[1] - 1) // block[1],
            1,
        )
        self.segment_sum_kernel(
            grid_sum,
            block,
            (
                per_token_max_view,
                # Stripe-relative offsets passed unchanged; the kernel
                # subtracts `offset_base` (the microbatch's first query
                # token) per call.
                self.query_section_offsets_device[
                    query_section_start : query_section_end + 1
                ],
                score_view,
                np.int32(query_section_count),
                np.int32(section_count),
                np.int32(self.per_token_max.shape[1]),
                np.int32(self.score_out.shape[1]),
                np.int32(query_token_start),
            ),
        )

        # Step 4 — top-k of (running ∪ tile) into the running buffer.
        score_view_t = torch.from_dlpack(score_view)
        self._merge_running_topk(
            section_count=section_count,
            candidate_start_i32=candidate_start_i32,
            query_section_start=query_section_start,
            query_section_end=query_section_end,
            query_section_count=query_section_count,
            score_view_t=score_view_t,
        )

    def _merge_running_topk(
        self,
        section_count: int,
        candidate_start_i32,
        query_section_start: int,
        query_section_end: int,
        query_section_count: int,
        score_view_t,
    ):
        """Pre-allocated 2-way top-k merge: stage running + tile values
        side-by-side into the combined buffer, then top-k + gather directly
        into the running buffer.
        """
        import cupy
        import torch

        keep = self.keep
        running_scores = self.topk_scores_t[query_section_start:query_section_end]
        running_indices = self.topk_indices_t[query_section_start:query_section_end]

        if section_count >= keep:
            torch.topk(
                score_view_t,
                k=keep,
                dim=1,
                largest=True,
                sorted=False,
                out=(
                    self.tile_values_t[:query_section_count],
                    self.tile_local_t[:query_section_count],
                ),
            )
            self.combined_scores_t[:query_section_count, :keep].copy_(running_scores)
            self.combined_scores_t[:query_section_count, keep:].copy_(
                self.tile_values_t[:query_section_count]
            )
            self.combined_indices_t[:query_section_count, :keep].copy_(running_indices)
            # Tile-local int64 → int32, lift to global by adding tile offset.
            self.combined_indices[:query_section_count, keep:] = (
                cupy.from_dlpack(self.tile_local_t[:query_section_count]).astype(
                    cupy.int32
                )
                + candidate_start_i32
            )
        else:
            # Tile narrower than keep: pad scores with -inf, indices with -1.
            self.combined_scores_t[:query_section_count, :keep].copy_(running_scores)
            self.combined_scores_t[
                :query_section_count, keep : keep + section_count
            ].copy_(score_view_t)
            self.combined_scores_t[
                :query_section_count, keep + section_count :
            ].fill_(float("-inf"))
            self.combined_indices_t[:query_section_count, :keep].copy_(running_indices)
            self.combined_indices[
                :query_section_count, keep : keep + section_count
            ] = cupy.arange(section_count, dtype=cupy.int32) + candidate_start_i32
            self.combined_indices_t[
                :query_section_count, keep + section_count :
            ].fill_(-1)

        # Final merge: top-k of (running ∪ tile). Scores write directly into
        # the running buffer; indices via torch.gather (int64 index).
        torch.topk(
            self.combined_scores_t[:query_section_count],
            k=keep,
            dim=1,
            largest=True,
            sorted=False,
            out=(running_scores, self.merge_pos_t[:query_section_count]),
        )
        torch.gather(
            self.combined_indices_t[:query_section_count],
            1,
            self.merge_pos_t[:query_section_count],
            out=running_indices,
        )

    def _maybe_log_progress(self, tile_idx: int, started: float):
        import cupy

        last = tile_idx + 1 == len(self.candidate_starts)
        if (tile_idx + 1) % 32 != 0 and not last:
            return
        self.compute_stream.synchronize()
        cupy.get_default_memory_pool().free_all_blocks()
        elapsed = time.monotonic() - started
        done = (tile_idx + 1) * self.candidate_tile_sections
        rate = done / max(elapsed, 1e-3) / 1e6
        print(
            f"{self.log_prefix}tile {tile_idx + 1}/{len(self.candidate_starts)} "
            f"elapsed {elapsed:.0f}s ({rate:.2f}M sect/s)",
            flush=True,
        )

    def _finalize(self) -> tuple[np.ndarray, np.ndarray]:
        import cupy

        sorted_order = cupy.argsort(-self.topk_scores, axis=1)
        sorted_scores = cupy.take_along_axis(self.topk_scores, sorted_order, axis=1)
        sorted_indices = cupy.take_along_axis(self.topk_indices, sorted_order, axis=1)
        query_global_ids = cupy.arange(
            self.stripe_start_section, self.stripe_end_section, dtype=cupy.int32
        )
        final_indices, final_scores = _drop_self_match(
            sorted_scores, sorted_indices, query_global_ids, self.num_neighbors
        )
        return cupy.asnumpy(final_indices), cupy.asnumpy(final_scores)

    def run(self) -> tuple[np.ndarray, np.ndarray]:
        import torch

        counts = [(0, 0), (0, 0)]
        for slot in range(min(2, len(self.candidate_starts))):
            counts[slot] = self._stage_tile(slot, self.candidate_starts[slot])

        started = time.monotonic()
        for tile_idx, candidate_start in enumerate(self.candidate_starts):
            slot = tile_idx % 2
            section_count, token_count = counts[slot]
            if section_count == 0:
                continue
            self.compute_stream.wait_event(self.copy_done[slot])
            candidate_start_i32 = np.int32(candidate_start)

            with self.compute_stream, torch.cuda.stream(self.torch_compute_stream):
                for query_section_start in range(
                    0, self.stripe_size, self.query_tile_sections
                ):
                    query_section_end = min(
                        query_section_start + self.query_tile_sections,
                        self.stripe_size,
                    )
                    self._score_microbatch(
                        slot,
                        section_count,
                        token_count,
                        candidate_start_i32,
                        query_section_start,
                        query_section_end,
                    )
                self.compute_done[slot].record(self.compute_stream)

            prefetch_idx = tile_idx + 2
            if prefetch_idx < len(self.candidate_starts):
                counts[slot] = self._stage_tile(
                    slot, self.candidate_starts[prefetch_idx]
                )

            self._maybe_log_progress(tile_idx, started)

        self.compute_stream.synchronize()
        return self._finalize()


def gt_stripe_maxsim(
    token_bank_host: np.ndarray,
    section_offsets_host: np.ndarray,
    stripe_start_section: int,
    stripe_end_section: int,
    num_neighbors: int,
    query_tile_sections: int,
    candidate_tile_sections: int,
    log_prefix: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact MaxSim top-k for sections in
    ``[stripe_start_section, stripe_end_section)`` against the whole section
    corpus. Returns ``(indices, scores)`` numpy arrays of shape
    ``(stripe_size, num_neighbors)``.

    ``token_bank_host`` is ``(T_total, dim)`` FP16; ``section_offsets_host``
    is ``(N+1,)`` int32 cumulative (section i's tokens live at rows
    ``[offsets[i] : offsets[i+1]]``).

    Caller picks the GPU via ``CUDA_VISIBLE_DEVICES``.
    """
    return _MaxSimStripeRunner(
        token_bank_host,
        section_offsets_host,
        stripe_start_section,
        stripe_end_section,
        num_neighbors,
        query_tile_sections,
        candidate_tile_sections,
        log_prefix,
    ).run()


def load_maxsim_corpus(
    model_root: Path, suffix: str
) -> tuple[np.ndarray, np.ndarray, list[CollectionShard], int]:
    """Walk ``*.{suffix}.sections.f16bin`` + ``*.sections.offsets.ibin``
    shards in canonical order. Returns
    ``(token_bank, section_offsets, shards, dimensions)``.

    ``section_offsets`` has shape ``(total_sections + 1,)`` int32 cumulative
    across all shards (section i's tokens live at
    ``token_bank[offsets[i] : offsets[i+1]]``). Each ``CollectionShard``'s
    ``row_count`` is its section count, ``row_offset`` the cumulative
    section count.
    """
    if not model_root.is_dir():
        raise FileNotFoundError(f"no model directory at {model_root}")
    shards: list[CollectionShard] = []
    cumulative_sections = 0
    cumulative_tokens = 0
    section_offsets_chunks: list[np.ndarray] = [np.zeros(1, dtype=np.int32)]
    token_chunks: list[tuple[int, int, Path]] = []
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
            offsets_blob = offsets_path
            with open(offsets_blob, "rb") as file:
                rows, _cols = struct.unpack("<II", file.read(8))
                local_offsets = np.frombuffer(
                    file.read(), dtype=np.int32, count=rows
                ).copy()
            n_sections = rows - 1
            shifted = (local_offsets + cumulative_tokens).astype(np.int32)
            # Local offsets already start at 0; drop the first element of
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
        raise FileNotFoundError(
            f"no `.{suffix}.sections.f16bin` files under {model_root}"
        )

    token_bank = np.empty((cumulative_tokens, dimensions), dtype=np.float16)
    for token_offset, token_count, path in token_chunks:
        blob = path
        with open(blob, "rb") as file:
            file.seek(8)
            destination = token_bank[token_offset : token_offset + token_count]
            file.readinto(memoryview(destination))  # type: ignore[arg-type]
    bad = ~np.isfinite(token_bank).all(axis=1)
    if bad.any():
        token_bank[bad] = 0
    section_offsets = np.concatenate(section_offsets_chunks).astype(np.int32)
    return token_bank, section_offsets, shards, dimensions


class MaxSimRetriever:
    """Brute-force exact MaxSim top-k for a multi-vector section corpus.

    Loads the full token bank + section offsets to one GPU at construction.
    ``search()`` accepts a query in the same ``(token_bank, offsets)``
    shape and runs matmul + segment-max + segment-sum + top-k against the
    resident corpus.

    For 60M sections × 128 dim (avg 3.4 tokens/section) the resident bank
    is ~52 GB FP16 — fits on H100 only after FP8/int8 quantization, or on
    B200 / multi-GPU for raw FP16.
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
        ) = load_maxsim_corpus(model_root, suffix)
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
        """``query_token_bank``: ``(T_q, dim)`` FP16. ``query_section_offsets``:
        ``(Q+1,)`` int32 cumulative. Returns ``(scores, indices)`` —
        both ``(Q, k)`` numpy arrays.
        """
        import cupy
        import torch

        if query_token_bank.ndim != 2 or query_token_bank.shape[1] != self.dimensions:
            raise ValueError(
                f"queries shape {query_token_bank.shape} != (?, {self.dimensions})"
            )
        n_queries = query_section_offsets.shape[0] - 1
        segment_max_kernel, segment_sum_kernel = _segment_kernels()

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
        segment_max_kernel(
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
        segment_sum_kernel(
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


# endregion
