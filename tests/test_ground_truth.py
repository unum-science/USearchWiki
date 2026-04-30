"""End-to-end synthetic test for ground_truth.py.

Generates a tiny multi-wiki, multi-shard tree of f16 embeddings, runs the full
script under --num-gpus 1, and validates outputs against a NumPy brute-force
reference. No real data needed.

Run:
  python -m pytest tests/test_ground_truth.py -s
or directly:
  python tests/test_ground_truth.py
"""

from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from wikiverse import write_bin, read_bin  # noqa: E402


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def build_synthetic_tree(
    root: Path,
    model_subdir: str,
    dimensions: int,
    wikis: dict[str, list[int]],
    seed: int = 0,
) -> tuple[np.ndarray, list[tuple[str, str, int, int]]]:
    """Create {root}/{model_subdir}/{wiki}/{stem}.body.f16bin tree.

    `wikis` maps wikiname -> list of shard sizes (row counts).
    Returns the concatenated ground-truth embeddings (in deterministic shard order)
    and a manifest of (wikiname, stem, row_offset, row_count).
    """
    rng = np.random.default_rng(seed)
    model_root = root / model_subdir
    all_rows: list[np.ndarray] = []
    manifest: list[tuple[str, str, int, int]] = []
    offset = 0
    for wikiname in sorted(wikis):
        wiki_dir = model_root / wikiname
        wiki_dir.mkdir(parents=True, exist_ok=True)
        for shard_index, row_count in enumerate(wikis[wikiname]):
            stem = f"000_{shard_index:05d}"
            raw = rng.standard_normal((row_count, dimensions)).astype(np.float32)
            normalized = normalize_rows(raw).astype(np.float16)
            write_bin(wiki_dir / f"{stem}.body.f16bin", normalized, dtype="f16")
            all_rows.append(normalized)
            manifest.append((wikiname, stem, offset, row_count))
            offset += row_count
    embeddings = np.concatenate(all_rows, axis=0) if all_rows else np.empty((0, dimensions), np.float16)
    return embeddings, manifest


def assemble_per_shard(
    model_root: Path, suffix: str, extension: str, dtype: str
) -> np.ndarray:
    """Read every `{wiki}/{stem}.{suffix}.ground_truth.{extension}` and
    concatenate in the same deterministic order discover_shards uses (sorted
    wikiname, then sorted stem)."""
    parts: list[np.ndarray] = []
    for wiki_dir in sorted(model_root.iterdir()):
        if not wiki_dir.is_dir():
            continue
        for path in sorted(wiki_dir.glob(f"*.{suffix}.ground_truth.{extension}")):
            parts.append(read_bin(path, dtype=dtype))
    return np.concatenate(parts, axis=0) if parts else np.empty((0, 0), dtype=np.float32)


def reference_topk(
    embeddings: np.ndarray, num_neighbors: int
) -> tuple[np.ndarray, np.ndarray]:
    """Brute-force exact top-k via NumPy float32 matmul, with self-match dropped."""
    f32 = embeddings.astype(np.float32)
    similarity = f32 @ f32.T
    total = embeddings.shape[0]
    # Drop self-match by setting diagonal to -inf, then take top-k.
    np.fill_diagonal(similarity, -np.inf)
    top_indices = np.argsort(-similarity, axis=1)[:, :num_neighbors].astype(np.int32)
    top_scores = np.take_along_axis(similarity, top_indices, axis=1).astype(np.float32)
    return top_indices, top_scores


def run_script(
    output_root: Path,
    model_subdir: str,
    dimensions: int,
    num_neighbors: int,
    query_tile_rows: int,
    candidate_tile_rows: int,
    num_gpus: int = 1,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "ground_truth.py"),
        "--output", str(output_root),
        "--model-subdir", model_subdir,
        "--dimensions", str(dimensions),
        "--output-suffix", "body",
        "--num-neighbors", str(num_neighbors),
        "--num-gpus", str(num_gpus),
        "--query-tile-rows", str(query_tile_rows),
        "--candidate-tile-rows", str(candidate_tile_rows),
    ]
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def compare_topk(
    expected_indices: np.ndarray,
    expected_scores: np.ndarray,
    actual_indices: np.ndarray,
    actual_scores: np.ndarray,
    score_tolerance: float = 5e-3,
) -> None:
    """Top-k may reorder ties, so compare as sets-of-(index, score-bucket) per row."""
    assert expected_indices.shape == actual_indices.shape, (
        f"shape mismatch: {expected_indices.shape} vs {actual_indices.shape}"
    )
    rows, k = expected_indices.shape
    mismatches: list[str] = []
    for row in range(rows):
        expected_set = set(expected_indices[row].tolist())
        actual_set = set(actual_indices[row].tolist())
        missing = expected_set - actual_set
        if missing:
            # If a missing expected index has a score essentially tied with an
            # actual index, that's a tie-break difference, not a real bug.
            tail_score_actual = float(actual_scores[row].min())
            for missing_index in missing:
                expected_pos = int(np.where(expected_indices[row] == missing_index)[0][0])
                expected_score = float(expected_scores[row, expected_pos])
                if abs(expected_score - tail_score_actual) > score_tolerance:
                    mismatches.append(
                        f"row {row}: expected idx {missing_index} (score "
                        f"{expected_score:.4f}) missing; actual tail score "
                        f"{tail_score_actual:.4f}"
                    )
                    break
        # No self-match in actual:
        if row in actual_set:
            mismatches.append(f"row {row}: self-match {row} present in actual top-k")
    if mismatches:
        raise AssertionError(
            f"{len(mismatches)} row mismatches; first 5:\n  "
            + "\n  ".join(mismatches[:5])
        )


def test_synthetic_end_to_end() -> None:
    dimensions = 64
    num_neighbors = 5
    wikis = {
        "alswiki": [120, 80],
        "rwwiki": [50],
        "simplewiki": [200, 30, 70],
    }
    with tempfile.TemporaryDirectory(prefix="gt_test_") as tmpdir:
        root = Path(tmpdir)
        embeddings, manifest = build_synthetic_tree(
            root, "tiny-model", dimensions, wikis, seed=42
        )
        total_vectors = embeddings.shape[0]
        print(f"synthetic corpus: {total_vectors} vectors x {dimensions} dim")

        # Use small tile sizes so the tiling logic gets exercised.
        completed = run_script(
            root,
            model_subdir="tiny-model",
            dimensions=dimensions,
            num_neighbors=num_neighbors,
            query_tile_rows=37,
            candidate_tile_rows=64,
        )
        print("--- script stdout (last 30 lines) ---")
        print("\n".join(completed.stdout.splitlines()[-30:]))

        model_root = root / "tiny-model"
        # Per-shard `.ground_truth.{ibin,fbin}` files; reassemble into a global matrix
        # using the same deterministic shard order the script uses.
        actual_indices = assemble_per_shard(model_root, "body", "ibin", "i32")
        actual_scores = assemble_per_shard(model_root, "body", "fbin", "f32")
        assert actual_indices.shape == (total_vectors, num_neighbors), actual_indices.shape
        assert actual_scores.shape == (total_vectors, num_neighbors), actual_scores.shape

        # No global file or manifest should be left behind.
        assert not (model_root / "ground_truth.body.ibin").exists()
        assert not (model_root / "ground_truth.body.manifest.json").exists()

        expected_indices, expected_scores = reference_topk(embeddings, num_neighbors)
        compare_topk(expected_indices, expected_scores, actual_indices, actual_scores)

        # Scratch dir cleaned up.
        scratch = model_root / "_ground_truth_scratch_body"
        assert not scratch.exists(), f"scratch dir not cleaned: {scratch}"

        assert (actual_scores[:, 0] <= 1.0 + 1e-3).all()
        for row in range(total_vectors):
            assert row not in set(actual_indices[row].tolist())

        print(f"PASS: {total_vectors} queries, k={num_neighbors}, all rows match")


def test_synthetic_multi_gpu_larger() -> None:
    """Bigger corpus across 4 GPUs with realistic-shaped tiles."""
    dimensions = 256
    num_neighbors = 20
    wikis = {
        "alswiki": [800, 1200],
        "rwwiki": [400],
        "simplewiki": [2000, 600, 1100],
        "enwiki": [1500, 1500],
    }
    with tempfile.TemporaryDirectory(prefix="gt_test_mgpu_") as tmpdir:
        root = Path(tmpdir)
        embeddings, _ = build_synthetic_tree(
            root, "tiny-mgpu", dimensions, wikis, seed=7
        )
        total_vectors = embeddings.shape[0]
        print(f"multi-gpu corpus: {total_vectors} vectors x {dimensions} dim")

        completed = run_script(
            root,
            model_subdir="tiny-mgpu",
            dimensions=dimensions,
            num_neighbors=num_neighbors,
            query_tile_rows=512,
            candidate_tile_rows=1024,
            num_gpus=4,
        )
        print("\n".join(completed.stdout.splitlines()[-15:]))

        model_root = root / "tiny-mgpu"
        actual_indices = assemble_per_shard(model_root, "body", "ibin", "i32")
        actual_scores = assemble_per_shard(model_root, "body", "fbin", "f32")
        assert actual_indices.shape == (total_vectors, num_neighbors)
        assert actual_scores.shape == (total_vectors, num_neighbors)

        expected_indices, expected_scores = reference_topk(embeddings, num_neighbors)
        compare_topk(expected_indices, expected_scores, actual_indices, actual_scores)
        print(f"PASS: {total_vectors} queries across 4 GPUs, all rows match")


if __name__ == "__main__":
    test_synthetic_end_to_end()
    test_synthetic_multi_gpu_larger()
