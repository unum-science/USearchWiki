---
license: apache-2.0
pretty_name: USearchWiki
language:
- multilingual
size_categories:
- 100M<n<1B
task_categories:
- feature-extraction
- sentence-similarity
- text-retrieval
source_datasets:
- HuggingFaceFW/finewiki
tags:
- embeddings
- wikipedia
- wikidata
- ann-benchmarks
- vector-search
- colbert
- usearch
- arxiv
- biorxiv
- medrxiv
configs:
- config_name: example
  data_files: data/example/parquet/??????????-??????????.parquet
- config_name: example-gte
  data_files: data/example/parquet/*.gte.parquet
- config_name: example-qwen
  data_files: data/example/parquet/*.qwen.parquet
- config_name: example-arctic
  data_files: data/example/parquet/*.arctic.parquet
- config_name: example-nomic
  data_files: data/example/parquet/*.nomic.parquet
- config_name: wikipedia
  data_files: data/wikipedia/parquet/??????????-??????????.parquet
- config_name: wikipedia-gte
  data_files: data/wikipedia/parquet/*.gte.parquet
- config_name: wikipedia-qwen
  data_files: data/wikipedia/parquet/*.qwen.parquet
- config_name: wikipedia-arctic
  data_files: data/wikipedia/parquet/*.arctic.parquet
- config_name: wikipedia-nomic
  data_files: data/wikipedia/parquet/*.nomic.parquet
- config_name: arxiv
  data_files: data/arxiv/parquet/??????????-??????????.parquet
- config_name: biorxiv
  data_files: data/biorxiv/parquet/??????????-??????????.parquet
- config_name: medrxiv
  data_files: data/medrxiv/parquet/??????????-??????????.parquet
- config_name: collections
  data_files: data/collections.parquet
---

# USearchWiki

An open, searchable atlas of wiki-style and scholarly text: every document is one row under a permanent key, with its sections, embeddings, cross-language links, dates, places and search graphs added beside the same rows.
It includes Wikipedia in 325 languages and the arXiv, bioRxiv and medRxiv preprint servers.

Start with `example`: the first 300'000 Wikipedia documents of a fixed shuffle, 6.1 GB, carrying every column, model file and graph the full collection does.

| Config      |  Documents |    Sections | Source                                     |
| :---------- | ---------: | ----------: | :----------------------------------------- |
| `example`   |    300'000 |     976'725 | the first documents of the full shuffle    |
| `wikipedia` | 60'354'903 | 197'268'428 | HuggingFace FineWiki, newest revisions     |
| `arxiv`     |  3'192'870 |  10'156'620 | 480'647 papers with text                   |
| `biorxiv`   |    352'284 |   2'043'412 | 79'113 papers with text                    |
| `medrxiv`   |     89'561 |     646'783 | 23'964 papers with text                    |

Each collection's plain config holds one row per document, and its `-gte`, `-qwen`, `-arctic` and `-nomic` configs hold one model's vectors, aligned row for row with it.
The `collections` config names every collection by the ordinal its keys carry.

## Loading

The shards are plain Parquet, and `pyarrow`, `pandas` and `dask` read every file directly:

```python
import pyarrow.parquet as pq

documents = pq.read_table("data/example/parquet/0000000000-0000300000.parquet")
vectors = pq.read_table("data/example/parquet/0000000000-0000300000.qwen.parquet")
```

The `datasets` library loads every config:

```python
from datasets import load_dataset

documents = load_dataset("unum-cloud/USearchWiki", "example", split="train")
```

Searching the vectors, including MaxSim over GTE sections, is what [USearch](https://github.com/unum-cloud/USearch) is for.

## Columns

Plain configs carry one row per document:

| Column           | Type            | Description                                  |
| :--------------- | :-------------- | :------------------------------------------- |
| `key`            | `binary(16)`    | UUIDv8 of collection, page and revision      |
| `collection`     | `utf8`          | Such as `enwiki` or `medrxiv`                |
| `title`, `url`   | `utf8`          | Versioned URL for preprints                  |
| `language`       | `utf8`          | Language code                                |
| `modified`       | `timestamp`     | Revision or posting time                     |
| `license`        | `utf8`          | SPDX identifier                              |
| `entity`         | `binary(16)`    | Wikidata entity, Wikipedia only              |
| `authors`        | list of `utf8`  | Preprints only                               |
| `categories`     | list of `utf8`  | Preprints only                               |
| `abstract`       | `utf8`          | Preprints, where the license allows          |
| `subject_dates`  | list of structs | Dates of the subject itself                  |
| `subject_places` | list of structs | Places of the subject itself                 |
| `text`           | `utf8`          | Rendered Markdown, null when withheld        |
| `sections`       | list of structs | Heading, level and byte range, tiling `text` |
| `links`          | list of structs | Span, section, entity, target, URL, context  |
| `dates`          | list of structs | Span, section and inclusive day range        |
| `places`         | list of structs | Span, section and bounding box in degrees    |
| `infoboxes`      | `utf8`          | FineWiki infobox JSON, Wikipedia only        |
| `markup`         | `utf8`          | `wikitext`, `latex` or `jats`                |
| `source`         | list of structs | The original files, verbatim                 |

Spans are UTF-8 byte ranges into the whole `text`, and a link with neither span nor section is a template, reference or back-matter link.
The `-gte` configs hold one 128-dimensional `float16` vector per section in `vectors`, and the dense configs hold `body` and `title` vectors per document, 1024-dimensional for Qwen3 and Arctic and 768-dimensional for Nomic.
Only `example` and `wikipedia` have model configs so far.

## Licensing

Wikipedia text is under CC BY-SA 4.0.
A preprint publishes its text, sections, annotations, markup and source files only under CC BY or CC0, and otherwise keeps its metadata with a null `text` and empty lists.
Abstracts follow the same rule for bioRxiv and medRxiv, while arXiv abstracts come from arXiv's own CC0 metadata.

## Caveats

FineWiki's text is the August 2025 snapshot, while the Wikidata and Wikimedia dumps behind the annotations are from October and November 2026, so a link to a page created or renamed in between may stay unresolved.
Ranges written with words, such as _from 1937 to 1939_, are kept as two separate dates.
Preprints carry no Wikidata entity, and their subject dates are the posting dates their servers report.
22.6% of arXiv papers have no text yet, being PDF-only, failing to parse or exceeding the conversion time limit; later releases add LaTeXML and OCR for them.
Preprint links are citations without Wikidata entities, and most preprint dates are the years of cited works.

## More

Pre-built USearch graphs, their recall, the key layout, the annotation rules and the full methodology live in the [GitHub repository](https://github.com/unum-cloud/USearchWiki).

## Citation

```bibtex
@dataset{Vardanian_USearchWiki,
  author = {Vardanian, Ash},
  title = {{USearchWiki: An Open, Searchable Atlas of Wiki-Style and Scholarly Text}},
  doi = {10.5281/zenodo.23977284},
  url = {https://github.com/unum-cloud/USearchWiki},
  license = {Apache-2.0}
}
```
