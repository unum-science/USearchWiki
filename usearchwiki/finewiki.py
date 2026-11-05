"""FineWiki source shards and the `.f16bin` files the embedding and ground-truth scripts wrote over them."""

from __future__ import annotations

import re
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow.parquet as pq

shard_pattern = re.compile(r"(\d{3})_(\d{5})\.parquet$")

numpy_dtypes = {"f16": np.float16, "f32": np.float32, "i32": np.int32}
file_extensions = {"f16": "f16bin", "f32": "fbin", "i32": "ibin"}

DType = Literal["f16", "f32", "i32"]

LFS_POINTER_PREFIX = b"version https://git-lfs"


@dataclass(frozen=True, slots=True)
class Shard:
    wikiname: str
    group: int
    index: int
    path: Path

    @property
    def stem(self) -> str:
        return f"{self.group:03d}_{self.index:05d}"


@dataclass(frozen=True, slots=True)
class CollectionShard:
    """A shard of an embedding collection, populated with its row count and the
    row offset that would assign each of its rows a deterministic global flat ID
    (used as the integer key for ground truth, USearch indexes, and any consumer
    that wants to map vectors back to their source article)."""

    wikiname: str
    stem: str
    path: Path
    row_offset: int
    row_count: int


def resolve_lfs_pointer(path: Path) -> Path:
    """If `path` is a Git-LFS pointer file, return the materialized blob in
    `.git/lfs/objects`; otherwise return the path unchanged.

    Pointer files are tiny ASCII stubs (~133 bytes) with an `oid sha256:<hex>`
    line. Repositories cloned with `GIT_LFS_SKIP_SMUDGE=1` (or with `git lfs
    fetch` only) keep the actual binaries under `.git/lfs/objects/<aa>/<bb>/<oid>`
    while the working tree holds pointers. Reading those blobs in place avoids
    checking out a duplicate copy of the dataset.
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
    # Walk parents of the resolved path so symlinked working trees still find
    # the original repo's `.git/lfs/objects`.
    for ancestor in path.resolve().parents:
        candidate = ancestor / ".git" / "lfs" / "objects" / oid[:2] / oid[2:4] / oid
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"{path}: LFS pointer references oid {oid} but no .git/lfs/objects/ contains it; "
        f"run `git lfs fetch` or pass --output to a tree that has the blobs"
    )


def discover_collection(
    model_root: Path, suffix: str = "body"
) -> list[CollectionShard]:
    """Walk an embedding collection's per-shard `.f16bin` files in canonical
    order (sorted wikiname, then sorted shard stem), reading each header to
    capture its row count, and return a list of `CollectionShard`s with
    cumulative `row_offset`s assigned.

    This is the *contract* for the global flat IDs: the i-th row of the n-th
    shard (after this canonical walk) gets ID `shards[n].row_offset + i`.
    Ground truth files, USearch index keys, and any consumer needing a
    deterministic vector-to-article mapping should reproduce this walk.
    """
    if not model_root.is_dir():
        raise FileNotFoundError(f"no model directory at {model_root}")
    shards: list[CollectionShard] = []
    cumulative = 0
    for wiki_dir in sorted(model_root.iterdir()):
        if not wiki_dir.is_dir():
            continue
        for path in sorted(wiki_dir.glob(f"*.{suffix}.f16bin")):
            stem = path.name[: -len(f".{suffix}.f16bin")]
            blob = resolve_lfs_pointer(path)
            with open(blob, "rb") as file:
                rows, _columns = struct.unpack("<II", file.read(8))
            shards.append(
                CollectionShard(
                    wikiname=wiki_dir.name,
                    stem=stem,
                    path=path,
                    row_offset=cumulative,
                    row_count=rows,
                )
            )
            cumulative += rows
    return shards


def find_snapshot(cache_dir: str | Path) -> Path:
    cache_dir = Path(cache_dir)
    snapshots = cache_dir / "datasets--HuggingFaceFW--finewiki" / "snapshots"
    if not snapshots.is_dir():
        raise FileNotFoundError(f"no FineWiki snapshot under {snapshots}")
    return max(snapshots.iterdir(), key=lambda path: path.stat().st_mtime)


def load_lang(cache_dir: str | Path, wikiname: str) -> list[Shard]:
    """Discover all FineWiki shards for a single language."""
    data_root = find_snapshot(cache_dir) / "data"
    wiki_dir = data_root / wikiname
    if not wiki_dir.is_dir():
        raise FileNotFoundError(f"no shard directory for {wikiname}")
    shards: list[Shard] = []
    for parquet_path in sorted(wiki_dir.glob("*.parquet")):
        match = shard_pattern.search(parquet_path.name)
        if not match:
            continue
        shards.append(
            Shard(
                wikiname=wikiname,
                group=int(match.group(1)),
                index=int(match.group(2)),
                path=parquet_path,
            )
        )
    return shards


def iter_articles(
    shard: Shard, text_column: str = "text", id_column: str = "id"
) -> Iterator[tuple[str, str]]:
    """Yield (id, text) tuples from a parquet shard, in parquet row order.

    Empty/null texts pass through as empty string — embedders should write
    zero vectors so row N in .f16bin aligns with parquet row N.
    """
    table = pq.read_table(shard.path, columns=[id_column, text_column])
    identifiers = table.column(id_column).to_pylist()
    texts = table.column(text_column).to_pylist()
    for identifier, text in zip(identifiers, texts, strict=True):
        yield str(identifier), text or ""


def load_shard_texts(
    shard: Shard, text_column: str = "text", id_column: str = "id"
) -> tuple[list[str], list[str]]:
    """Load a whole shard into parallel lists (ids, texts), parquet row order preserved."""
    identifiers: list[str] = []
    texts: list[str] = []
    for identifier, text in iter_articles(
        shard, text_column=text_column, id_column=id_column
    ):
        identifiers.append(identifier)
        texts.append(text)
    return identifiers, texts


def write_bin(path: str | Path, matrix: np.ndarray, dtype: DType) -> None:
    """Write a 2-D matrix to a binary file with a `uint32 rows, uint32 cols` header."""
    if matrix.ndim != 2:
        raise ValueError(f"expected 2-D matrix, got shape {matrix.shape}")
    matrix = np.ascontiguousarray(matrix.astype(numpy_dtypes[dtype], copy=False))
    rows, columns = matrix.shape
    with open(path, "wb") as file:
        file.write(struct.pack("<II", rows, columns))
        file.write(matrix.tobytes(order="C"))


def read_bin(path: str | Path, dtype: DType) -> np.ndarray:
    """Read a 2-D matrix from a binary file with a `uint32 rows, uint32 cols` header."""
    with open(path, "rb") as file:
        rows, columns = struct.unpack("<II", file.read(8))
        return np.frombuffer(file.read(), dtype=numpy_dtypes[dtype]).reshape(
            rows, columns
        )


def shard_filename(stem: str, dtype: DType) -> str:
    return f"{stem}.{file_extensions[dtype]}"
