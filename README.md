# USearchWiki

USearchWiki is an open, searchable atlas of wiki-style and scholarly text.
Every document is one row under a permanent key, with its sections inside it, and everything learned about that text — embeddings from several models, links resolved across languages, the dates and places it mentions — is added beside the same rows, together with the indexes that search them.
New collections, models and annotations extend the atlas without renumbering what is already there.

It lets a person, a retrieval pipeline or an agent ask:

- _"Find the passages closest to this paragraph, in any language, and score whole documents by all of their sections."_
- _"Which sections link to the entity Paris, whatever the language of the article?"_
- _"Show the passages that mention a date between 1939 and 1945 and a place within 200 km of Warsaw."_
- _"Rank these documents under four different embedding models and compare where they disagree."_

The atlas includes Wikipedia in every language, whose layout and keys every other collection follows, and the arXiv, bioRxiv and medRxiv preprint servers:

| Collection  |  Documents |    Sections |  With text |
| :---------- | ---------: | ----------: | ---------: |
| `wikipedia` | 60'354'903 | 197'268'428 |        all |
| `example`   |    300'000 |     976'725 |        all |
| `arxiv`     |  3'192'870 |  10'156'620 |    480'647 |
| `biorxiv`   |    352'284 |   2'043'412 |     79'113 |
| `medrxiv`   |     89'561 |     646'783 |     23'964 |

`wikipedia` holds the newest revisions of FineWiki's 325 language editions, and the preprint collections are in English.

The `example` collection is the one to start with: 6.1 GB that carries every column, model file and graph of the full Wikipedia, in a uniform sample of its languages.

Every row carries its own license as an SPDX identifier, and what a row publishes follows it.
Wikipedia text is under CC BY-SA 4.0.
A preprint publishes its text, sections, annotations, markup and source files only under CC BY or CC0; every other preprint keeps its metadata — title, authors, categories, dates, URL and license — with a null `text` and empty lists.
Abstracts follow the same rule for bioRxiv and medRxiv, while arXiv abstracts come from arXiv's own CC0 metadata.

Every document of the current release carries:

- __Text__: the rendered Markdown of the whole document, with its sections as byte ranges that tile it exactly.
- __Links__: each link as a byte span, the Wikidata entity it names and the document it resolves to, both keyed the same way in every language.
- __Dates and places__: every date range and coordinate box found in the text, flagged by whether a template, a linked entity, a text pattern or the collection's metadata produced it.
- __Source__: the original files verbatim, from wikitext and JATS XML to the TeX of a paper.
- __Embeddings__: one vector per document from three dense models, and one vector per section from a late-interaction model for MaxSim search.
- __Indexes__: USearch graphs over the section and document vectors, each reaching 99% recall on held-out queries.

Preprint collections have no model files or graphs yet.

## Dataset Structure

Each collection holds `parquet/` with one row per document, cut into 1M-row stems named by their first and end rows, and its graphs beside it.
Files sharing a stem are row-aligned, one per model, so vectors need no join and any model can be fetched or dropped alone.
Sizes below use GB = 2³⁰ bytes.

```sh
.
├── data
│   ├── collections.parquet                          # collection names by ordinal, frozen and append-only
│   ├── example                                      # 6.1 GB, the first 300'000 shuffled Wikipedia documents
│   │   └── ...                                      # the same shape as wikipedia
│   ├── wikipedia
│   │   ├── index-gte-sections.safegraphs            # 97.0 GB, every section under its document's key, for MaxSim
│   │   ├── index-qwen-documents.safegraphs          # 130.4 GB, one body vector per document
│   │   ├── index-arctic-documents.safegraphs        # 130.4 GB, one body vector per document
│   │   └── parquet                                  # 61 stems of each kind
│   │       ├── 0000000000-0001000000.parquet        # 3.8 GB, text, sections, links, dates, places, source
│   │       ├── 0000000000-0001000000.gte.parquet    # 0.7 GB, one 128-d vector per section
│   │       ├── 0000000000-0001000000.qwen.parquet   # 3.5 GB, body and title vectors
│   │       ├── 0000000000-0001000000.arctic.parquet # 3.5 GB, body and title vectors
│   │       ├── 0000000000-0001000000.nomic.parquet  # 2.6 GB, body and title vectors
│   │       ├── ...
│   │       └── 0060000000-0060354903.nomic.parquet
│   ├── arxiv
│   │   └── parquet                                  # one row per paper, in progress
│   ├── biorxiv
│   │   └── parquet
│   │       └── 0000000000-0000352284.parquet        # 3.1 GB, every paper, text for 79'113
│   └── medrxiv
│       └── parquet
│           └── 0000000000-0000089561.parquet        # 0.7 GB, every paper, text for 23'964
└── README.md
```

Wikipedia documents are shuffled once with a fixed seed, so any prefix of the stems is a uniform sample across languages, and `example` is exactly the first 300'000 of them.

### Document Rows

```sh
>>> import pyarrow.parquet as pq
>>> pq.read_schema('data/wikipedia/parquet/0000000000-0001000000.parquet')

key: fixed_size_binary[16] not null      # the document, see Keys
collection: dictionary<string> not null  # such as enwiki, arxiv, biorxiv or medrxiv
title: string not null
url: string not null                     # versioned for preprints
language: string                         # such as en, null when unknown
modified: timestamp[ms] not null         # UTC, the revision or posting date
license: dictionary<string> not null     # SPDX, such as CC-BY-SA-4.0, CC-BY-4.0 or CC0-1.0
entity: fixed_size_binary[16]            # the Wikidata entity, Wikipedia only
authors: list<string>
categories: list<string>
abstract: large_string                   # where the license allows
subject_dates: list<struct<start: int32, end: int32, property: uint32, precision: uint8, calendar: uint8, source: uint8>>
subject_places: list<struct<south: float, west: float, north: float, east: float, precision_deg: float, property: uint32, source: uint8>>
text: large_string                       # rendered Markdown, null when withheld
sections: list<struct<heading: string, level: uint8, start: uint32, end: uint32>>
links: list<struct<span: struct<start: uint32, end: uint32>, section: uint16, entity: fixed_size_binary[16], target: fixed_size_binary[16], url: string, context: uint8>>
dates: list<struct<span: struct<start: uint32, end: uint32>, section: uint16, start: int32, end: int32, property: uint32, precision: uint8, calendar: uint8, source: uint8>>
places: list<struct<span: struct<start: uint32, end: uint32>, section: uint16, south: float, west: float, north: float, east: float, precision_deg: float, property: uint32, source: uint8>>
infoboxes: large_string                  # FineWiki's infobox JSON, Wikipedia only
markup: dictionary<string>               # wikitext, latex or jats
source: list<struct<path: string, content: large_binary>>
```

- `sections` tile `text` exactly: each is a UTF-8 byte range `[start, end)`, the first starts at 0, every next one starts where the previous ended, and the last ends with `text`.
  `level` is 0 for the lead and otherwise the count of `#`, and `heading` is null for the lead.
- Every `span` in `links`, `dates` and `places` is a UTF-8 byte range into the whole `text`, and `section` is the index of the section holding it.
  A null span with a section marks a fact annotated in that section with no matching words, such as a date carried by a linked entity whose link text is a name.
  A link with neither span nor section is a template, reference or back-matter link, which has no rendered position.
- A link's `entity` is the Wikidata entity its target names, `target` the key of the document it resolves to, and `url` an external address, each null when absent.
  Entities and targets are UUIDs that hold across languages: a link to _Paris_ from any wiki carries the same entity.
  `context` says where the link sat in the markup: 0 prose, 1 table, 2 template, 3 reference, 4 other, and null when unknown.
- Dates are inclusive day ranges since 1970 in the proleptic Gregorian calendar, with astronomical years below 1, so `1965` is the whole year and `14 July 1789` a single day.
  `precision` follows Wikidata: 7 century, 8 decade, 9 year, 10 month, 11 day.
  `calendar` is 0 for Gregorian, 1 for a value converted from Julian, 2 when the source calendar is unknown.
  `property` is the Wikidata property a linked entity contributed, such as 580 for a start time, and 0 otherwise.
- Places are boxes in degrees, and a point has equal pairs; `west > east` crosses the antimeridian.
  `precision_deg` is the source precision and 0 when absent.
- The `source` field of a date or place says how it was found: 0 a date or coordinate template, 1 a linked Wikidata entity, 2 a text pattern, 3 the primary and 4 a secondary `geo_tags` coordinate, and 5 the collection's metadata, such as the posting date of a preprint.
  In `subject_dates` and `subject_places`, source 1 is the document's own Wikidata entity.
- `subject_dates` and `subject_places` describe the subject itself: Wikidata birth, death, inception and coordinates, `geo_tags`, title coordinates and infobox templates for an article, and posting dates for a preprint.
  `dates` and `places` describe only what the text mentions.
- The `source` column holds the original files verbatim under their `path`: one entry of wikitext or JATS XML, or every TeX file of an arXiv paper.

Schema metadata records the `collection`, the `schema_version`, the `license_policy`, the `key` layout, and for Wikipedia the `finewiki_commit`, `wikidata_dump_date` and `sql_dump_date`.

### Model Files

Each model has its own file per stem, row-aligned with the main file and starting with the same `key` column:

| File               | Columns         | Type                                                |
| :----------------- | :-------------- | :-------------------------------------------------- |
| `*.gte.parquet`    | `vectors`       | `list<fixed_size_list<halffloat, 128>>`, by section |
| `*.qwen.parquet`   | `body`, `title` | `fixed_size_list<halffloat, 1024>`                  |
| `*.arctic.parquet` | `body`, `title` | `fixed_size_list<halffloat, 1024>`                  |
| `*.nomic.parquet`  | `body`, `title` | `fixed_size_list<halffloat, 768>`                   |

A GTE cell holds one vector for each of its document's `sections`, in the same order.

### Keys

Every `key` is an RFC 9562 UUIDv8 of 16 big-endian bytes, and the same bytes identify the row in its model files and every graph.
Its 122 custom bits hold `collection:16, page:32, revision:36, section:16, sentence:22`, placed around the fixed version and variant fields, so sorting keys groups a collection's pages and a page's revisions.
`section` and `sentence` are reserved for addressing parts of a document and are 0 in every stored key.
An entity key uses collection `0xFFFF` and the Wikidata QID number as its page.

```py
def decode(key: bytes) -> tuple[int, int, int, int]:
    value = int.from_bytes(key, "big")
    payload = (value >> 80) << 74 | ((value >> 64) & 0xFFF) << 62 | value & ((1 << 62) - 1)
    return payload >> 106, (payload >> 74) & 0xFFFF_FFFF, (payload >> 38) & 0xF_FFFF_FFFF, (payload >> 22) & 0xFFFF
```

`collection` is the ordinal in `collections.parquet`: 0 to 324 are the Wikipedia language editions, 325 is arXiv, 326 bioRxiv and 327 medRxiv.
`page` and `revision` carry each collection's own identifiers:

| Collection       | Identifier                    | `page`                                | `revision`             |
| :--------------- | :---------------------------- | :------------------------------------ | :--------------------- |
| Wikipedia        | page and revision IDs         | page ID                               | revision ID            |
| bioRxiv, medRxiv | `10.1101/2024.06.04.24308466` | `24308466`                            | `days << 8 \| version` |
| arXiv            | `YYMM.NNNNN`                  | `yymm · 100'000 + number`             | version                |
| arXiv            | `archive/YYMMNNN`             | `10⁹ + archive index · 10⁷ + yymmnnn` | version                |

For bioRxiv and medRxiv, `days` counts from 1970 to the posting date in a dated DOI, and is 0 for an undated one.
A Wikipedia key opens its article at `https://<lang>.wikipedia.org/?curid=<page>&oldid=<revision>`.

### Search Graphs

The graphs are [SafeGraphs](https://github.com/unum-cloud/usearch) checkpoints built with USearch at connectivity 64 and an insertion expansion of 256, over the `inner` metric on unit vectors, which USearch reports as `dot`.
The GTE graph holds every section under its document's key, so a plain search returns the document of the nearest section, and USearch's MaxSim scores whole documents: every query section takes its best match within a document, and those maxima sum.
The last 10'000 documents of the shuffle are held out of every graph as queries.

Recall is 1-recall@10 over the held-out queries against exact scans of the first 1'000 of them, measured on a 96-core Graviton4 with every graph in RAM:

| Graph                        | Insert, s | Refine, s | Expansion | Queries/s | Rows/s | Recall@10 |
| :--------------------------- | --------: | --------: | --------: | --------: | -----: | --------: |
| `index-gte-sections`, MaxSim |     2'118 |     1'618 |       256 |    16'055 | 52'823 |     0.995 |
| `index-qwen-documents`       |     4'145 |         … |       128 |    32'416 | 32'416 |     0.990 |
| `index-arctic-documents`     |     3'997 |     3'955 |       512 |     9'393 |  9'393 |     0.993 |

The GTE row queries USearch's MaxSim with every section of a held-out document and ranks documents; the dense rows query one body vector per document by inner product, which on these unit vectors ranks like angular distance.
Each graph searches at least as many rows per second as it inserted, counting refinement, so none was bought with a slower search.
Served in place from one NVMe drive instead of RAM, every graph keeps its recall at the same expansion: 0.996 for GTE, 0.990 for Qwen3 and 0.993 for Arctic.

## Installation

Install straight from GitHub with [uv](https://github.com/astral-sh/uv), no clone required.

```sh
uv pip install "git+https://github.com/unum-cloud/USearchWiki.git"                     # read the shards
uv pip install "usearchwiki[search] @ git+https://github.com/unum-cloud/USearchWiki.git"  # search the graphs with USearch v3
```

To hack on the code, clone it and install editable instead.

```sh
git clone https://github.com/unum-cloud/USearchWiki.git
cd USearchWiki
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[search]"
```

The `dense`, `ground` and `colbert` extras reproduce the embeddings and the original ground truth, and need GPUs.

## Usage

The dataset lives in the public `usearch-wiki` bucket on AWS S3 in `us-west-2`, readable anonymously.
Download the example collection, 6.1 GB with every column, model file and graph:

```sh
mkdir -p data/example
aws s3 cp --no-sign-request s3://usearch-wiki/data/collections.parquet data/
aws s3 sync --no-sign-request s3://usearch-wiki/data/example/ data/example/
```

If you need just one model's vectors of the full Wikipedia:

```sh
aws s3 sync --no-sign-request s3://usearch-wiki/data/wikipedia/parquet/ data/wikipedia/parquet/ \
    --exclude "*" --include "*.qwen.parquet"
```

The `usearchwiki` package reads every collection lazily, row group by row group:

```py
from usearchwiki import WikiDataset, civil_from_days

data = WikiDataset.open("data/example")
document = data.document(219)  # "Channel wing", with its sections, links, dates and places
for date in document.dates:
    section = document.sections[date["section"]]
    print(section.heading, document.quote(date["span"]), civil_from_days(date["start"]))

data.vectors("qwen", 219, 1)              # (1, 1024) float16
data.vectors("gte", 219, 1)               # (7, 128) float16, one row per section
for key, score in data.search_like(219, "gte", 10):  # MaxSim over all of its sections, needs usearch>=3
    print(score, data.find(key).title)
```

`python -m usearchwiki data/example 219` prints one document with every annotation.


## Methodology

The dataset is built in stages, each verified before the next, by tools kept outside this repository.

### Dataset Sources

[HuggingFaceFW/finewiki](https://huggingface.co/datasets/HuggingFaceFW/finewiki) — August 2025 snapshot, 325 languages, 61.5M articles, pinned to commit `8bd13e72e6a002407649b3e898535f42ceb1aeb9`.

FineWiki is extracted from Wikimedia's __Enterprise HTML dumps__ (not raw wikitext), so templates are fully rendered by MediaWiki's own engine.
This avoids the well-known content loss that plagues wikitext-based parsers, as `mwparserfromhell` cannot expand templates.
Section headings, tables, math, and lists are preserved as Markdown.
Bot-generated stubs, disambiguation pages, and cross-language leakage are filtered out.

FineWiki holds several revisions of about 1.2M pages.
USearchWiki keeps only the newest revision of each page, which is how 61.5M source rows become 60'354'903 articles.

The annotations come from three more sources:

- __Wikidata__: the [`all.json.bz2` dump](https://dumps.wikimedia.org/wikidatawiki/entities/) of 2026-10-26, for entities, sitelinks, dates, coordinates and template families.
- __Wikimedia SQL dumps__: [`page`, `redirect` and `geo_tags`](https://dumps.wikimedia.org/backup-index.html) of every wiki, from October and November 2026, for title resolution and article coordinates.
- __Site information__: each wiki's `siteinfo-namespaces` dump, for the localized File and Category prefixes that are not links.

The preprint collections come from their servers directly.
bioRxiv and medRxiv contribute the metadata of every paper and, under CC BY or CC0, its abstract and full-text JATS XML.
arXiv contributes its CC0 metadata for every paper, from arXiv's bulk metadata snapshot, and, under CC BY or CC0, the TeX sources from its bulk source archive.
An arXiv paper's text is its TeX converted to Markdown by pandoc, math kept as TeX, with the title and abstract as its lead section; the original files stay verbatim in `source`.

No chunking is used for the dense models.
Short-content models only process the abstract.
Long-context models are prioritized and receive the whole document in the original form.
Sections split the rendered text at its Markdown headings, so the sections of an article concatenate back to its FineWiki `text` byte for byte.

### Corpus Structure

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



Measured by scanning every parquet shard, including rendered Markdown `text` and raw `wikitext` columns:

| Quantity                                         |    Total | Per article |
| :----------------------------------------------- | -------: | ----------: |
| Articles                                         |   61.55M |           … |
| Rendered text bytes: `text` column               | 195.2 GB |      3.2 KB |
| Wikitext bytes: `wikitext` column                | 337.1 GB |      5.5 KB |
| Markdown paragraphs: blank-line-separated blocks |   254.2M |        4.13 |
| Section headings: `#`, `##`, `###`               |   206.3M |        3.35 |

35% of articles have a single paragraph (stubs), 65% have ≤ 3, only 10% have 8+.
Paragraph length distribution:

- 12%: under 50 bytes, mostly headings or one-liners,
- 38%: in 200–800 bytes, the prose sweet spot,
- 4%: over 3.2 KB, long lists or tables rendered as one block.

Annotation density extracted from raw `wikitext`:

| Annotation kind                           |     Total | Articles touched |
| :---------------------------------------- | --------: | ---------------: |
| Plain `[[wikilinks]]`                     |    1.42 B |            99.4% |
| Templates `{{...}}`                       |   0.998 B |            98.6% |
| Piped links `[[T\|d]]`                    |   0.648 B |            89.3% |
| Citations `<ref>...`                      |   0.400 B |            71.0% |
| External URLs `[https://...]`             |       84M |            41.8% |
| Categories `[[Category:...]]`             |       55M |            16.3% |
| Tables `{...}`                            |       19M |            14.6% |
| Files / images `[[File:...]]`             |       14M |             7.2% |
| __Section anchors `[[Article#Section]]`__ | __11.5M__ |         __6.4%__ |
| Math `<math>...`                          |      6.4M |             0.5% |
| Self anchors `[[#Section]]`               |      2.6M |             0.7% |
| Galleries `<gallery>`                     |      2.5M |             3.2% |
| Inline interwiki `[[lang:...]]`           |     0.83M |             1.1% |

Section anchors deserve attention: they form a 11.5M-edge __paragraph-level link graph__ already curated by editors — a built-in supervision signal for sub-article retrieval evaluation.

### Embedding Models

Each model embeds the same article corpus independently.
No chunking is applied — short-context models see truncated articles, long-context models see the full text.
Dense models produce one vector per article, for its body and separately for its title.
GTE-ModernColBERT produces one vector per token, and those are mean-pooled within each section, so the dataset stores one 128-dimensional vector per section rather than per token.

| Model                                                                                 | Type              | Dims | Context | Params | Reported score |
| :------------------------------------------------------------------------------------ | :---------------- | ---: | ------: | -----: | :------------- |
| [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)              | Dense, decoder    | 1024 |    32 K |  600 M | 70.7 MTEB v2   |
| [GTE-ModernColBERT-v1](https://huggingface.co/lightonai/GTE-ModernColBERT-v1)         | ColBERT, encoder  |  128 |  8-32 K |  139 M | 88.4 LongEmbed |
| [arctic-embed-l-v2.0](https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0) | Dense, encoder    | 1024 |     8 K |  568 M | 55.6 BEIR      |
| [nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)        | Dense, encoder    |  768 |     8 K |  137 M | 62.3 MTEB v1   |
| [e5-mistral-7b-instruct](https://huggingface.co/intfloat/e5-mistral-7b-instruct)      | Dense, decoder    | 4096 |     4 K |  7.1 B | 66.6 MTEB v1   |

Every model is under Apache 2.0 except e5-mistral-7b-instruct, under MIT.
Qwen3-Embedding builds on Alibaba's Qwen3 and e5-mistral on Mistral-7B, fine-tuned by Microsoft; GTE-ModernColBERT is LightOn's tuning of Answer.AI's ModernBERT, arctic-embed is Snowflake's tuning of BAAI's BGE-M3 over Meta's XLM-R, and nomic-embed builds on NomicBERT.
E5-Mistral is planned and not yet embedded.

All vectors are stored as `float16`, which NumPy, PyArrow and Parquet read natively, unlike `bfloat16`.
Encoder models ran in [TEI](https://github.com/huggingface/text-embeddings-inference) and decoder models in [vLLM](https://github.com/vllm-project/vllm) with `--task embed`.
`embed_articles.py` sends each article's first 16'384 characters, which the server truncates again to the model's context, and GTE-ModernColBERT mean-pools its token vectors within each section.
The scripts that produced the vectors and the original GPU ground truth stay in this repository as provenance: `embed_articles.py`, `embed_sections.py`, `late_chunking.py` and `ground_truth.py`.

### Pipeline

1. FineWiki shards and the embeddings are pulled and checked against their SHA-256 sums.
2. An audit proves that section spans match every GTE offset file, that `(wiki, page)` is unique, and that every vector is unit length.
3. One seed-42 shuffle and the key encoding are frozen before anything else is written.
4. The Wikidata JSON dump is streamed once into compact tables of sitelinks, dates and places.
5. Each article's wikitext is parsed for links, templates and dates, and its rendered text is split into sections and annotated.
6. Graph inputs are cut from the vectors in shuffled order, and each graph is built, then refined until it reaches 99% recall.

Every section concatenates back to its FineWiki text, every non-null link `target` exists among the document keys, every span falls inside its text on a UTF-8 boundary, and every model file's keys equal its stem's keys.

### Annotation Quality

Wikipedia sections hold 1'304'896'451 links: 89.3% name a Wikidata entity, 86.2% resolve to an article, and 0.79% could not be placed in their section.
They carry 504'197'545 dates, 72.8% from patterns, 26.3% from linked entities and 1.0% from templates, and 318'427'995 places, 99.1% of them from linked entities.
`subject_dates` and `subject_places` add 35'833'701 dates and 43'388'134 places about the articles' subjects, 47.6% of the places from `geo_tags`.

Links come from the wikitext, matched to the rendered sections by heading text and in order, never by count.
A link is placed only where its display words appear in order after the previous link of the same section; a free search would misplace the 23% of English display strings that repeat within a section.
Template and reference links rarely appear in the rendered text, so they are kept without a span or section rather than at a wrong position.
Titles resolve through FineWiki, then the page and redirect tables, and an entity comes from the target article or from current Wikidata sitelinks; 68'478 stored `wikidata_id` values disagreed with Wikidata and were corrected.

Dates and places come from three sources, each flagged:

- Templates: date and coordinate templates found in each wiki through their Wikidata items, such as `{{Birth date}}`, `{{Start date}}`, `{{Date}}` and `{{Coord}}`, so localized names are covered without a per-wiki list.
- Linked entities: a linked item that is a calendar item, or has a point in time, start or end time, a coordinate location, or a complete bounding box.
  Birth dates and inceptions of linked items are excluded, since every link to a person or a city would otherwise become a date.
- Patterns: per-wiki rules for years, full dates, ranges, decades, centuries and eras in the 20 largest wikis, and ISO dates everywhere.
  Ranges written with words, such as _from 1937 to 1939_, stay as two separate dates.
  Coordinates need degrees with a hemisphere, since a bare degree sign is mostly a temperature.

Wikidata values use best rank only, skip unknown and missing values, and convert Julian dates through Julian day numbers.
Coordinates at exactly 0°, 0° are placeholders and are dropped, as are coordinates on other globes.

### Limits and Caveats

__Not every arXiv paper has text yet.__
Of the 3'192'870 papers, 77.4% convert from TeX to Markdown, 10.5% fail to parse, 8.0% are PDF-only submissions, 2.7% exceed the conversion time limit, and 1.5% have no recognizable LaTeX main file or fail otherwise.
Such a paper keeps its metadata with a null `text`; under CC BY or CC0 it still publishes its TeX in `source`, while a PDF-only paper has an empty `source`.
Of the 620'870 openly licensed papers, 140'223 have no text: 55'795 PDF-only, 53'035 parse failures, 28'260 timeouts and 3'133 others.
Later releases close the gap with LaTeXML, the converter behind arXiv's own HTML pages, for papers pandoc cannot parse, a longer time limit, and OCR of the PDF-only papers.
Four papers in the source archive have no metadata and are left out.

__Preprint annotations are thin yet.__
Preprint links are citations and web links, so none names a Wikidata entity: 35% carry a DOI or arXiv URL, and 7.2% of arXiv links resolve to another document of the atlas, under 1% for bioRxiv and medRxiv.
Over 90% of preprint dates are bare years, mostly those of cited works, and coordinates are almost absent.
Later releases link the molecules, genes, species, theorems, datasets and other entities named in preprint text to Wikidata, and tell citation years apart from the dates a paper is about.

__An arXiv paper is the version its source archive holds.__
The bulk archive keeps the version current when each monthly tar was cut, so a paper revised later carries that earlier version in its key and URL.

__The sources are 14 months apart.__
FineWiki's text is the August 2025 snapshot, while Wikidata is the 2026-10-26 dump and the page, redirect and `geo_tags` tables come from dumps of October and November 2026.
A link to a page created or renamed in between may stay unresolved, and an entity may carry facts newer than the article text.

__Three Nomic vectors were recomputed.__
Three source rows held a `NaN`: `elwiki` page _Βραβεία Nickelodeon Kids' Choice_, `zhwiki` page _第五届全国人民代表大会代表名单_ and `hewiki` page _פרס היצירה לסופרים עבריים_.
They were re-embedded on a CPU with the original settings, which reproduce 36 neighboring rows at cosine 1.0000.

__An article may lack an entity.__
The 747'741 articles without a Wikidata item, 1.2% of the total, have a null `entity`, and links to them carry targets without entities.

## Enrichment

USearchWiki goes beyond embeddings.
Wikipedia articles contain rich structure — hyperlinks with descriptive anchor text, section hierarchies, tables, cross-lingual alignments, disambiguation — that can be parsed directly from the wikitext markup without any model inference.
Combined with Wikidata properties and model-generated QA pairs, this creates __multiple independent notions of relevance__ for the same corpus.
The link spans, entities, sections, dates and places above implement the first three ideas and the temporal and geographic parts of the fifth; the rest are planned.

### Anchor Text as Queries

Wikipedia's internal links carry __anchor text__ that describes the target article in context.
FineWiki preserves the raw wikitext alongside cleaned text, so every `[[target|anchor]]` link is extractable:

```
Wikitext:  ...played [[football (soccer)|football]] between the [[Home Nations]]...
Extracted: "football"                → [[Football (soccer)]]
           "Home Nations"            → [[Home Nations]]
           "England"                 → [[England national football team]]
           "tour of Central Europe"  → [[1907–08 in English football#National team]]
```

Each anchor is a natural short query for which the target article is ground truth.
English Wikipedia has __~200M internal links__ — this creates millions of _(query, relevant\_document)_ pairs for free, far more realistic than synthetic queries or self-recall.
Different articles link to the same target with different anchors, capturing diverse search intents for the same document.
In this dataset every such pair is a link's `span` and its `target` key.

__What it enables:__
Retrieval benchmarks where the query is natural human-written text, not a randomly sampled vector.
Evaluates whether each embedding model's vector space places the anchor text near the target article.

### Cross-Lingual Alignment

Via Wikidata sitelinks, the same concept exists across 2–300 languages.
The article about "Paris" in English (`Q90`) maps to "パリ" in Japanese, "Париж" in Russian, "巴黎" in Chinese.
FineWiki provides `wikidata_id` for most articles, and this dataset corrects it against current Wikidata and stores it as the `entity` key, so the articles sharing an `entity` are one concept in every language:

```
enwiki: { title: "Paris", entity: Q90, ... }
jawiki: { title: "パリ",   entity: Q90, ... }
ruwiki: { title: "Париж", entity: Q90, ... }
```

__What it enables:__
Cross-lingual retrieval ground truth: given a query in language A, find the correct article in language B.
Tests whether multilingual embedding models (Qwen3, Arctic) properly align their vector spaces across scripts.
No existing ANN benchmark tests cross-lingual retrieval at scale.

### Section Hierarchy

FineWiki preserves section headings as Markdown `##`/`###`.
A section heading is a natural query for its content:

```markdown
## Early life                → biographical passage about childhood
## Discovery and naming      → how the element was found
## Economic impact           → financial analysis section
```

Each article becomes a tree of passages at different granularities — the full article, individual sections, and individual paragraphs.
The heading itself serves as a query for which the section content is ground truth.

__What it enables:__
Passage-level retrieval within articles — a more granular task than full-article retrieval.
Tests whether embedding models capture topic boundaries within long documents.

### Disambiguation Negatives

Disambiguation pages explicitly list articles sharing names but meaning different things.
From wikitext: `[[Mercury (planet)]]`, `[[Mercury (element)]]`, `[[Mercury (mythology)]]`, `[[Freddie Mercury]]`.

These are natural __hard negatives__ — articles that share surface-level query terms but should retrieve different documents.
Extractable from disambiguation templates in wikitext and from Wikidata `instance_of` divergence.

__What it enables:__
Precision benchmarks: can the embedding model distinguish "Mercury" the planet from "Mercury" the element when the query is ambiguous?
Standard ANN benchmarks have no concept of hard negatives.

### Filtered Search

Wikidata properties and Wikipedia page statistics provide rich, __naturally correlated__ filter attributes.
Unlike synthetic random labels used in NeurIPS'23, these filters correlate with semantic content — articles about nearby cities cluster both geographically AND in embedding space.

| Attribute           | Type          | Source                 | Example filter                        |
| :------------------ | :------------ | :--------------------- | :------------------------------------ |
| `instance_of`       | Categorical   | Wikidata P31           | "only biographies" or "only cities"   |
| `country`           | Categorical   | Wikidata P17           | "articles about France"               |
| `coordinates`       | Geo (lat/lon) | Wikidata P625          | "within 200 km of Tokyo"              |
| `inception_year`    | Temporal      | Wikidata P571 / P569   | "events between 1939 and 1945"        |
| `population`        | Numeric       | Wikidata P1082         | "cities with population > 500K"       |
| `pageviews_monthly` | Numeric       | Wikimedia pageviews    | "popular articles only (> 10K views)" |
| `num_sitelinks`     | Numeric       | Wikidata sitelinks     | "exists in 50+ languages"             |
| `num_inlinks`       | Numeric       | `pagelinks` SQL dump   | "over 1,000 incoming links"           |
| `language`          | Categorical   | FineWiki `in_language` | "only Japanese articles"              |

The coordinates and dates are stored today as `subject_places`, `subject_dates`, `places` and `dates`; the other attributes are planned.

__What it enables:__
Filtered ANN benchmarks with realistic, multi-type filter conjunctions: "French cities founded after 1800 with population > 100K."
Controllable selectivity: ship query sets at 0.1%, 1%, 10%, 50% selectivity.

### Citation Co-Occurrence

Two articles citing the same external source (URL or DOI in `<ref>` tags) are related in a way distinct from semantic similarity or link structure.
Articles about the same scientific finding, or different news coverage of the same event, share citations but may not link to each other.

```
Wikitext:  <ref>{{cite journal|doi=10.1038/nature12373|...}}</ref>
Extracted: doi:10.1038/nature12373 → shared by articles A, B, C
```

__What it enables:__
A "citation graph" orthogonal to the link graph — tests whether embeddings capture source-level relatedness.

### Generated QA Pairs

Run a small model (Qwen-2.5-3B) on each article to generate 3–5 factual questions whose answers are in the text.
The generated question becomes a query, the source article becomes the ground truth document.

```
Article: "The 1908 Olympics, held in London, featured..."
Generated: "Where were the 1908 Olympics held?" → this article
```

__What it enables:__
Standard retrieval evaluation in the BEIR/MSMARCO format, but over Wikipedia's full breadth.

### Infobox Facts

FineWiki extracts infoboxes as structured JSON, kept in the `infoboxes` column:

```json
[{"title": "1907–08 British Home Championship",
  "data": {"Host country": "England, Ireland, Scotland and Wales",
           "Dates": "15 February – 11 April 1908",
           "Teams": "4"}},
 {"title": "Tournament statistics",
  "data": {"Top scorer(s)": "Vivian Woodward · Jimmy Quinn (4 goals)"}}]
```

Each key-value pair is a structured fact that can be turned into a query:
"Who was the top scorer of the 1907–08 British Home Championship?" → this article.

__What it enables:__
Structured knowledge retrieval — queries derived from real structured data, not synthetic.

## Related Work

Cohere's Wikipedia releases are the closest analogue, embedding Wikipedia passages in many languages with one Cohere model per release.
They ship the vectors and the passage text, but no links, dates or places and no search graphs, and a different model means a different download.

BEIR, MTEB and KILT measure retrieval quality against curated relevance labels, and KILT grounds its tasks in a single Wikipedia snapshot.
Their corpora are small enough to search exactly, so they test models rather than search engines.

The big-ann-benchmarks tracks scale to a billion vectors, but each dataset comes from one model, and most from image or word descriptors older than current text encoders.

USearchWiki joins the two: several current embedding models over the same multilingual rows, with graphs at full scale, and with links, entities, dates and places to filter and label those rows.

## Citation

If USearchWiki helps your research or product, please cite it:

```bibtex
@dataset{Vardanian_USearchWiki,
  author = {Vardanian, Ash},
  title = {{USearchWiki: An Open, Searchable Atlas of Wiki-Style and Scholarly Text}},
  doi = {10.5281/zenodo.23977284},
  url = {https://github.com/unum-cloud/USearchWiki},
  license = {Apache-2.0}
}
```

A machine-readable [`CITATION.cff`](https://github.com/unum-cloud/USearchWiki/blob/main/CITATION.cff) is provided at the repository root.

## License

The code in this repository is licensed under [Apache 2.0](https://github.com/unum-cloud/USearchWiki/blob/main/LICENSE).

Dataset licensing depends on the components:

- __Wikipedia text__: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- __Preprint text__: each paper's own license, named in its `license` column; text, sections, annotations, markup and source files are published only under [CC BY](https://creativecommons.org/licenses/by/4.0/) or [CC0](https://creativecommons.org/publicdomain/zero/1.0/)
- __Preprint metadata__: titles, authors, categories, dates and URLs of every paper, published whatever its license; arXiv's abstracts come from its CC0 metadata
- __FineWiki extraction__: [Apache 2.0](https://huggingface.co/datasets/HuggingFaceFW/finewiki)
- __Embeddings__: Governed by each model's license (see table above — all selected models use Apache 2.0 or MIT)
- __Annotations__: Derived from Wikimedia and Wikidata dumps ([CC0](https://creativecommons.org/publicdomain/zero/1.0/) for Wikidata, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) for Wikipedia)
