"""Embed FineWiki shards via a running TEI server.

Usage:
  python embed.py --cache-dir /path/to/hf-cache --output /path/to/embeddings \\
      --wiki enwiki --model-subdir qwen3-embedding-0.6b --dimensions 1024

For title embeddings, add: --text-column title --output-suffix title --char-cap 256
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

import httpx
import numpy as np

from wikiverse import Shard, load_lang, load_shard_texts, write_bin


def select_shards(all_shards: list[Shard], gpu_id: int, world_size: int) -> list[Shard]:
    return [
        shard for index, shard in enumerate(all_shards) if index % world_size == gpu_id
    ]


async def embed_texts(
    client: httpx.AsyncClient,
    url: str,
    texts: list[str],
    batch_size: int,
    concurrency: int,
    character_cap: int,
    dimensions: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Embed `texts` via TEI in input order; empty inputs map to zero rows."""
    semaphore = asyncio.Semaphore(concurrency)
    output = np.zeros((len(texts), dimensions), dtype=np.float16)

    nonempty_indices = [index for index, text in enumerate(texts) if text]
    truncated = [texts[index][:character_cap] for index in nonempty_indices]
    received: list[list[float] | None] = [None] * len(truncated)

    async def submit(start: int) -> None:
        end = min(start + batch_size, len(truncated))
        chunk = truncated[start:end]
        async with semaphore:
            response = await client.post(
                url, json={"inputs": chunk, "truncate": True}, timeout=600.0
            )
            response.raise_for_status()
            vectors = response.json()
        for offset, vector in enumerate(vectors):
            received[start + offset] = vector

    started = time.monotonic()
    if truncated:
        await asyncio.gather(
            *(submit(index) for index in range(0, len(truncated), batch_size))
        )
    elapsed = time.monotonic() - started

    for received_index, vector in enumerate(received):
        output[nonempty_indices[received_index]] = np.asarray(vector, dtype=np.float16)

    return output, {
        "embed_seconds": elapsed,
        "n_docs": len(texts),
        "n_nonempty": len(truncated),
    }


async def run(args: argparse.Namespace) -> None:
    output_root = Path(args.output) / args.model_subdir
    output_root.mkdir(parents=True, exist_ok=True)

    suffix = f".{args.output_suffix}" if args.output_suffix else ""

    shards = load_lang(args.cache_dir, args.wiki)
    owned = select_shards(shards, args.gpu_id, args.world_size)
    pending = [
        shard
        for shard in owned
        if not (output_root / shard.wikiname / f"{shard.stem}{suffix}.f16bin").exists()
    ]
    print(
        f"[gpu{args.gpu_id} TEI col={args.text_column} suffix='{args.output_suffix}'] "
        f"{len(owned)} owned, {len(pending)} pending @ {args.url}",
        flush=True,
    )
    if not pending:
        return

    started = time.monotonic()
    total_docs = 0

    async with httpx.AsyncClient() as probe:
        try:
            response = await probe.get(
                args.url.rsplit("/", 1)[0] + "/health", timeout=10.0
            )
            response.raise_for_status()
        except Exception as error:
            raise SystemExit(f"TEI not reachable at {args.url}: {error}")

    async with httpx.AsyncClient() as client:
        for shard in pending:
            wiki_dir = output_root / shard.wikiname
            wiki_dir.mkdir(parents=True, exist_ok=True)
            vector_path = wiki_dir / f"{shard.stem}{suffix}.f16bin"

            _identifiers, texts = load_shard_texts(shard, text_column=args.text_column)
            if not texts:
                np.zeros((0, args.dimensions), dtype=np.float16).tofile(vector_path)
                continue

            embeddings, stats = await embed_texts(
                client,
                args.url,
                texts,
                batch_size=args.batch_size,
                concurrency=args.concurrency,
                character_cap=args.character_cap,
                dimensions=args.dimensions,
            )
            if embeddings.shape != (len(texts), args.dimensions):
                raise RuntimeError(
                    f"shape {embeddings.shape} != ({len(texts)}, {args.dimensions})"
                )

            temp_path = vector_path.with_suffix(vector_path.suffix + ".tmp")
            write_bin(temp_path, embeddings, dtype="f16")
            temp_path.rename(vector_path)

            total_docs += stats["n_docs"]
            docs_per_second = stats["n_docs"] / max(stats["embed_seconds"], 1e-3)
            print(
                f"[gpu{args.gpu_id} TEI] {shard.wikiname}/{shard.stem}{suffix}: "
                f"{stats['n_docs']} docs ({stats['n_nonempty']} non-empty) "
                f"in {stats['embed_seconds']:.1f}s -> {docs_per_second:.0f} doc/s",
                flush=True,
            )

    wall_seconds = time.monotonic() - started
    print(
        f"[gpu{args.gpu_id} TEI] DONE: {total_docs} docs in {wall_seconds:.0f}s "
        f"-> {total_docs/max(wall_seconds,1):.0f} doc/s",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="/home/ubuntu/wikiverse-data/hf-cache")
    parser.add_argument("--output", default="/home/ubuntu/wikiverse-data/embeddings")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument(
        "--wiki", required=True, help="single language code (enwiki, dewiki, etc)"
    )
    parser.add_argument("--url", default="http://localhost:8080/embed")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="docs per HTTP request"
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--text-column", default="text", choices=["text", "title"])
    parser.add_argument(
        "--character-cap",
        type=int,
        default=16384,
        help="truncate each input at this many characters (~max_length × 4 for English)",
    )
    parser.add_argument(
        "--output-suffix",
        default="body",
        help="filename suffix, e.g. 'body' or 'title'",
    )
    parser.add_argument(
        "--model-subdir",
        default="qwen3-embedding-0.6b",
        help="output goes to {output}/{model-subdir}/, e.g. snowflake-arctic-embed-l-v2.0",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=1024,
        help="embedding dimensionality (1024 Qwen3/arctic, 768 nomic, 4096 e5-mistral)",
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
