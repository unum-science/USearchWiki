# WikiVerse

Multi-model embedding dataset built on [HuggingFace FineWiki](https://huggingface.co/datasets/HuggingFaceFW/finewiki), designed for approximate nearest neighbor (ANN) search benchmarking with [USearch](https://github.com/unum-cloud/usearch) and other vector search engines.

The same Wikipedia corpus — chunked, cleaned, and enriched with graph metadata — is embedded by multiple models spanning dense encoders, decoder-based LLMs, and late-interaction (ColBERT-style) architectures.
Each model's embeddings ship with precomputed ground-truth k-nearest neighbors, enabling reproducible recall and throughput benchmarks without re-running expensive exact search.

## Why WikiVerse?

Existing ANN benchmarks suffer from three gaps:

1. __Stale descriptors.__
   The most popular benchmarks (SIFT-1B, Deep-1B, GloVe) use features from 2014-2021 — image descriptors and word vectors, not modern text embeddings.
2. __Single-model datasets.__
   Each benchmark is produced by one model.
   You cannot compare how the _same_ retrieval engine handles different vector distributions without re-embedding.
3. __No decoder embeddings.__
   State-of-the-art embedding models (GTE-Qwen, Llama-Embed-Nemotron, Qwen3-Embedding) are decoder-based LLMs, yet no ANN benchmark uses their outputs.

WikiVerse fixes all three: one corpus, multiple models, modern architectures, with graph-structured metadata for filtered search.

## Source Corpus

[HuggingFaceFW/finewiki](https://huggingface.co/datasets/HuggingFaceFW/finewiki) — August 2025 snapshot, 325 languages, 61.5M articles.

FineWiki is extracted from Wikimedia's __Enterprise HTML dumps__ (not raw wikitext), so templates are fully rendered by MediaWiki's own engine.
This avoids the well-known content loss that plagues wikitext-based parsers, as `mwparserfromhell` cannot expand templates.
Section headings, tables, math, and lists are preserved as Markdown.
Bot-generated stubs, disambiguation pages, and cross-language leakage are filtered out.

### Text processing

No chunking is used.
Short-content models only process the abstract.
Long-context models are prioritized and receive the whole document in the original form.

### Scale

| Scope                     | Articles | Parquet, GB | Avg Bytes/Article |
| :------------------------ | -------: | ----------: | ----------------: |
| English                   |     6.6M |          38 |             5,700 |
| Top 5: EN, DE, FR, ES, RU |    15.7M |          86 |             5,460 |
| Top 10 by text volume¹    |    22.9M |         120 |             5,250 |
| Top 20                    |    41.6M |         149 |             3,580 |
| All 325 languages         |    61.6M |        ~170 |             2,740 |

¹ EN, DE, FR, ES, RU, IT, JA, ZH, PL, UK — excluding bot-generated wikis (Cebuano, Swedish, Waray, Egyptian Arabic) which inflate article counts with minimal text.

Parquet weight includes both `text` and `wikitext` columns; pure text is roughly half.
Average bytes/article drops at wider scope because smaller wikis are dominated by stubs.

## Embedding Models

Each model embeds the same article corpus independently.
No chunking is applied — short-context models see truncated articles, long-context models see the full text.
Dense models produce one vector per article.
ColBERT models produce one vector per token (~2,000 vectors per average article).

| Model                                                                                 | Year | Type              | Dims | Context |  Params | License    | Base / Fine-tuned by                     | Perf           |
| :------------------------------------------------------------------------------------ | ---: | :---------------- | ---: | ------: | ------: | :--------- | :--------------------------------------- | :------------- |
| [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)              | 2025 | Dense (decoder)   | 1024 |    32 K |   600 M | Apache 2.0 | Qwen3 (Alibaba)                          | 70.7 MTEB v2   |
| [GTE-ModernColBERT-v1](https://huggingface.co/lightonai/GTE-ModernColBERT-v1)         | 2025 | ColBERT (encoder) |  128 |  8-32 K |   139 M | Apache 2.0 | ModernBERT (Answer.AI) / LightOn         | 88.4 LongEmbed |
| [arctic-embed-l-v2.0](https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0) | 2024 | Dense (encoder)   | 1024 |     8 K | 303 M ¹ | Apache 2.0 | XLM-R (Meta) → BGE-M3 (BAAI) / Snowflake | 55.6 BEIR      |
| [nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)        | 2024 | Dense (encoder)   |  768 |     8 K |   137 M | Apache 2.0 | NomicBERT (Nomic)                        | 62.3 MTEB v1   |
| [e5-mistral-7b-instruct](https://huggingface.co/intfloat/e5-mistral-7b-instruct)      | 2023 | Dense (decoder)   | 4096 |     4 K |   7.1 B | MIT        | Mistral-7B (Mistral AI) / Microsoft      | 66.6 MTEB v1   |

¹ 568M total, 303M active (non-embedding parameters).

### Compute Estimates

All embeddings stored in FP16.
Decoder models use [vLLM](https://github.com/vllm-project/vllm) with `--task embed`.
Encoder models use [TEI](https://github.com/huggingface/text-embeddings-inference) with the Hopper Docker image.
FP8 quantization can improve throughput ~1.5× with negligible quality loss.

Token counts vary by tokenizer — CJK text produces ~1 token per 2-3 bytes, Latin/Cyrillic ~1 per 4-5 bytes.
Average article length across all languages is ~400 tokens, but this is dragged down by millions of stubs in smaller wikis; English articles average ~2,700 tokens.

| Model                  | Throughput | Total tokens |   Time | Vectors | Storage | Notes                          |
| :--------------------- | ---------: | -----------: | -----: | ------: | ------: | :----------------------------- |
| Qwen3-Embedding-0.6B   |  500 doc/s |         24 B |  1.4 d |  61.6 M |  126 GB | Full articles                  |
| GTE-ModernColBERT-v1   |  800 doc/s |         24 B |  0.9 d |  24.3 B |  6.2 TB | ~400 token vectors per article |
| arctic-embed-l-v2.0    |  800 doc/s |         28 B |  0.9 d |  61.6 M |  126 GB | Truncated at 8K tokens         |
| nomic-embed-text-v1.5  | 1200 doc/s |         21 B |  0.6 d |  61.6 M |   95 GB | Truncated at 8K tokens         |
| e5-mistral-7b-instruct |   50 doc/s |         21 B | 14.3 d |  61.6 M |  505 GB | Truncated at 4K tokens         |

> Single H100 80 GB, full dataset — 61.6M articles, all 325 languages.

## Metadata Enrichment

### Compute Estimates

## Dataset Layout

Layout mirrors [FineWiki's](https://huggingface.co/datasets/HuggingFaceFW/finewiki) `data/<wiki>/<group>_<shard>.parquet` structure: one directory per Wikipedia language, with shard filenames preserved 1:1.
Each `.f16bin` is row-aligned with its source parquet — `.f16bin` row N is the embedding of parquet row N, in native order.
If the source text was empty or null the row is a zero vector (`norm == 0`); the parquet's `id` column provides the doc identifier, so no separate ids file is needed.

Binary format: `u32` rows count, `u32` columns count, then `rows × cols` little-endian `f16` values — directly compatible with [USearch](https://github.com/unum-cloud/USearch)'s and the Big-ANN benchmark ecosystem.

`.body.f16bin` is the article-body embedding; `.title.f16bin` is the title-only embedding (short-context, useful for title-vs-body retrieval studies).

```
unum-cloud/WikiVerse/
├── README.md
├── LICENSE
├── .gitattributes
├── wikiverse.py                              # consumer module: load_lang, read_bin, ...
├── embed.py                                  # reference embedding pipeline (TEI-based)
│
├── qwen3-embedding-0.6b/                     # 1024-dim, decoder, float16
│   ├── enwiki/
│   │   ├── 000_00000.body.f16bin             # mirrors enwiki/000_00000.parquet
│   │   ├── 000_00000.title.f16bin
│   │   ├── 000_00001.body.f16bin
│   │   ├── 000_00001.title.f16bin
│   │   └── ...
│   ├── dewiki/
│   │   └── ...
│   └── ...                                   # one dir per Wikipedia language
│
├── snowflake-arctic-embed-l-v2.0/            # 1024-dim, encoder, float16
│   └── <wiki>/<group>_<shard>.{body,title}.f16bin
│
├── nomic-embed-text-v1.5/                    #  768-dim, encoder, float16
│   └── <wiki>/<group>_<shard>.{body,title}.f16bin
│
├── e5-mistral-7b-instruct/                   # 4096-dim, decoder, float16 (planned)
│   └── <wiki>/<group>_<shard>.{body,title}.f16bin
│
└── gte-moderncolbert-v1/                     #  128-dim per token, ColBERT (planned)
    └── <wiki>/<group>_<shard>.{body,title}.f16bin
```

Binary embedding files are tracked via [Git LFS](https://git-lfs.com).
To clone the repo without downloading the binaries (~600 GB):

```sh
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/unum-cloud/WikiVerse
```

Load embeddings for a single language (Python):

```python
from wikiverse import read_bin
mat = read_bin("qwen3-embedding-0.6b/enwiki/000_00000.body.f16bin", dtype="f16")
# mat.shape == (n_articles_in_shard, 1024)
```

Or pull just one model's embeddings for a single language:

```sh
huggingface-cli download unum-cloud/WikiVerse \
    --repo-type dataset \
    --include "qwen3-embedding-0.6b/enwiki/*"
```

### Workflow

The embedding pipeline is designed for multi-day runs on GPU servers with checkpoint/resume:

```sh
# 1. Download FineWiki articles
python corpus.py --lang en --output corpus/

# 2. Embed with each model (resume-safe — rerun after interruptions)
python embed.py --model qwen3-0.6b --input corpus/ --output embeddings/ --resume
python embed.py --model e5-mistral-7b --input corpus/ --output embeddings/ --resume
python embed.py --model arctic-embed-l-v2 --input corpus/ --output embeddings/ --resume
python embed.py --model nomic-v1.5 --input corpus/ --output embeddings/ --resume
python embed.py --model gte-moderncolbert --input corpus/ --output embeddings/ --resume

# 3. Extract graph metadata
python graph.py --lang en --output graph/

# 4. Compute ground truth for each model
python ground_truth.py --embeddings embeddings/qwen3-0.6b/ --k 100 --queries 10000
python ground_truth.py --embeddings embeddings/e5-mistral-7b/ --k 100 --queries 10000

# 5. Upload to HuggingFace
python upload.py --repo unum-cloud/WikiVerse
```

Each step is idempotent.
Progress is tracked in `state/*.json` files — if a job dies (OOM, SSH drop, GPU error), rerunning the same command picks up from the last checkpoint.
Adding a new embedding model requires only step 2 + step 4 — the corpus and graph are shared.

## Hosting

| Location                                                                           | Storage/mo (1 TB) | Egress/GB | Notes                                                            |
| ---------------------------------------------------------------------------------- | ----------------- | --------- | ---------------------------------------------------------------- |
| [HuggingFace Hub](https://huggingface.co/unum-cloud/WikiVerse)                     | Free              | Free      | Primary. Xet storage, unlimited public downloads                 |
| [AWS S3](https://aws.amazon.com/s3/pricing/) Standard                              | $23.00            | $0.09     | S3-compatible mirror. Egress adds up fast for popular datasets   |
| [Nebius Object Storage](https://docs.nebius.com/object-storage/resources/pricing/) | $15.05            | $0.015    | S3-compatible. ~35% cheaper storage, ~6× cheaper egress than AWS |

## License

The embedding pipeline code in this repository is licensed under [Apache 2.0](LICENSE).

Dataset licensing depends on the components:

- __Wikipedia text__: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- __FineWiki extraction__: [Apache 2.0](https://huggingface.co/datasets/HuggingFaceFW/finewiki)
- __Embeddings__: Governed by each model's license (see table above — all selected models use Apache 2.0 or MIT)
- __Graph metadata__: Derived from Wikimedia/Wikidata dumps ([CC0](https://creativecommons.org/publicdomain/zero/1.0/) for Wikidata, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) for Wikipedia)
