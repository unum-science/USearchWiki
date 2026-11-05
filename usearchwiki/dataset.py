"""Reads the published USearchWiki layout: document shards, their model files and graphs.

A collection folder such as `data/example` or `data/medrxiv` holds `parquet/` with one row per document,
cut into stems named by their first and end rows, and the graphs beside it.
Every row is keyed by a UUIDv8 that names its collection, page and revision.

Example usage:
    dataset = WikiDataset.open("data/example")
    document = dataset.document(42)
    print(document.title, [section.heading for section in document.sections])
    for key, score in dataset.search_like(42, "qwen", 10):
        print(score, dataset.find(key).title)
"""

from __future__ import annotations

import bisect
import datetime
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ENTITY_COLLECTION = 0xFFFF
"""The `collection` field of a Wikidata entity key, whose `page` field is then the QID number."""

GRAPHS = {
    "gte": "index-gte-sections",
    "qwen": "index-qwen-documents",
    "arctic": "index-arctic-documents",
}
"""Graph files beside `parquet/`; GTE holds every section under its document's key, for MaxSim."""

DIMENSIONS = {"gte": 128, "qwen": 1024, "arctic": 1024, "nomic": 768}

SOURCES = {
    0: "template",
    1: "linked entity",
    2: "pattern",
    3: "primary geo tag",
    4: "secondary geo tag",
    5: "collection metadata",
}
"""How a date or place was found, by the `source` field of its struct."""

LINK_CONTEXTS = {0: "prose", 1: "table", 2: "template", 3: "reference", 4: "other"}
"""Where a link sat in the source markup, by its `context` field."""


@dataclass(frozen=True, slots=True)
class WikiKey:
    """The 122 payload bits of a row key: collection 16, page 32, revision 36, section 16 and sentence 22."""

    collection: int
    page: int
    revision: int
    section: int = 0
    sentence: int = 0

    @staticmethod
    def decode(key: bytes) -> WikiKey:
        value = int.from_bytes(key, "big")
        if (value >> 76) & 0xF != 8 or (value >> 62) & 0x3 != 2:
            raise ValueError(f"not a UUIDv8: {key.hex()}")
        payload = (value >> 80) << 74 | ((value >> 64) & 0xFFF) << 62 | value & ((1 << 62) - 1)
        return WikiKey(
            collection=payload >> 106,
            page=(payload >> 74) & 0xFFFF_FFFF,
            revision=(payload >> 38) & 0xF_FFFF_FFFF,
            section=(payload >> 22) & 0xFFFF,
            sentence=payload & 0x3F_FFFF,
        )

    def encode(self) -> bytes:
        payload = (
            self.collection << 106
            | self.page << 74
            | self.revision << 38
            | self.section << 22
            | self.sentence
        )
        value = (payload >> 74) << 80 | 0x8 << 76 | ((payload >> 62) & 0xFFF) << 64 | 0x2 << 62
        return (value | payload & ((1 << 62) - 1)).to_bytes(16, "big")

    @property
    def article(self) -> WikiKey:
        """The key of the whole document: section and sentence zeroed."""
        return WikiKey(self.collection, self.page, self.revision)

    @property
    def qid(self) -> int | None:
        return self.page if self.collection == ENTITY_COLLECTION else None


def civil_from_days(days: int) -> tuple[int, int, int]:
    """Proleptic Gregorian `(year, month, day)` of a day count since 1970, astronomical years below 1."""
    days += 719_468
    era = (days if days >= 0 else days - 146_096) // 146_097
    day_of_era = days - era * 146_097
    year_of_era = (
        day_of_era - day_of_era // 1460 + day_of_era // 36_524 - day_of_era // 146_096
    ) // 365
    day_of_year = day_of_era - (365 * year_of_era + year_of_era // 4 - year_of_era // 100)
    shifted_month = (5 * day_of_year + 2) // 153
    day = day_of_year - (153 * shifted_month + 2) // 5 + 1
    month = shifted_month + 3 if shifted_month < 10 else shifted_month - 9
    return year_of_era + era * 400 + (month <= 2), month, day


@dataclass(slots=True)
class Section:
    """A heading and the UTF-8 byte range `[start, end)` it covers in its document's text."""

    heading: str | None
    level: int
    start: int
    end: int
    utf8: bytes = field(repr=False)
    """The whole document text as UTF-8, shared by the document and all its sections."""

    @property
    def text(self) -> str:
        return self.utf8[self.start : self.end].decode()


@dataclass(slots=True)
class Document:
    row: int
    key: WikiKey
    collection: str
    title: str
    url: str
    language: str | None
    modified: datetime.datetime
    license: str
    entity: int | None
    authors: list[str]
    categories: list[str]
    abstract: str | None
    subject_dates: list[dict]
    """Facts about the subject itself, each `{start, end, property, precision, calendar, source}`."""
    subject_places: list[dict]
    """Each `{south, west, north, east, precision_deg, property, source}`; a point has equal pairs."""
    text: str | None
    """Null when the licence withholds it, and then every list below is empty."""
    sections: list[Section]
    links: list[dict]
    """Each `{span, section, entity, target, url, context}`, with a QID and a document key or None."""
    dates: list[dict]
    """Each `{span, section, start, end, property, precision, calendar, source}`, in days since 1970."""
    places: list[dict]
    """Each `{span, section, south, west, north, east, precision_deg, property, source}`."""
    infoboxes: str | None
    markup: str | None
    source: list[dict]
    """Each `{path, content}`: an original file verbatim, such as the wikitext or the JATS XML."""
    utf8: bytes = field(repr=False)

    def quote(self, span: dict | None) -> str | None:
        """The text a span covers, decoded from the UTF-8 bytes it counts."""
        if span is None:
            return None
        return self.utf8[span["start"] : span["end"]].decode()


def _stems(folder: Path) -> list[tuple[int, int, Path]]:
    """Main files of `folder` as `(first, end, path)`, ordered by their first row, without model files."""
    stems = []
    for path in folder.glob("*.parquet"):
        name, _, rest = path.name.partition(".")
        if rest != "parquet":
            continue
        first, _, end = name.partition("-")
        stems.append((int(first), int(end), path))
    return sorted(stems)


def _key_or_none(raw: bytes | None) -> WikiKey | None:
    return WikiKey.decode(raw) if raw else None


class WikiDataset:
    """One collection folder, read lazily: rows are fetched by row group, vectors and graphs on request."""

    def __init__(self, folder: Path, collections: list[str]):
        self.folder = folder
        self.collections = collections
        self.stems = _stems(folder / "parquet")
        self._graphs: dict[tuple[str, str], object] = {}

    @staticmethod
    def open(folder: str | Path) -> WikiDataset:
        folder = Path(folder)
        collections = pq.read_table(
            folder.parent / "collections.parquet", columns=["ordinal", "name"]
        ).sort_by("ordinal")
        return WikiDataset(folder, collections["name"].to_pylist())

    def __len__(self) -> int:
        return self.stems[-1][1] if self.stems else 0

    def _rows(self, suffix: str, first: int, count: int, columns: list[str] | None) -> pa.Table:
        """Rows `[first, first + count)` of the `suffix` files, reading only the row groups they touch."""
        pieces = []
        index = bisect.bisect_right([stem_first for stem_first, _, _ in self.stems], first) - 1
        while count > 0:
            stem_first, stem_end, path = self.stems[index]
            file = pq.ParquetFile(path.with_name(path.name.replace(".parquet", suffix)))
            start, end = first - stem_first, min(first - stem_first + count, stem_end - stem_first)
            group_rows = file.metadata.row_group(0).num_rows
            groups = list(range(start // group_rows, (end - 1) // group_rows + 1))
            table = file.read_row_groups(groups, columns=columns)
            pieces.append(table.slice(start - groups[0] * group_rows, end - start))
            first, count, index = first + end - start, count - (end - start), index + 1
        return pa.concat_tables(pieces)

    def document(self, row: int) -> Document:
        cells = self._rows(".parquet", row, 1, None).to_pylist()[0]
        utf8 = (cells["text"] or "").encode()
        return Document(
            row=row,
            key=WikiKey.decode(cells["key"]),
            collection=cells["collection"],
            title=cells["title"],
            url=cells["url"],
            language=cells["language"],
            modified=cells["modified"],
            license=cells["license"],
            entity=cells["entity"] and WikiKey.decode(cells["entity"]).qid,
            authors=cells["authors"] or [],
            categories=cells["categories"] or [],
            abstract=cells["abstract"],
            subject_dates=cells["subject_dates"],
            subject_places=cells["subject_places"],
            text=cells["text"],
            sections=[Section(**section, utf8=utf8) for section in cells["sections"]],
            links=[
                {
                    **link,
                    "entity": link["entity"] and WikiKey.decode(link["entity"]).qid,
                    "target": _key_or_none(link["target"]),
                }
                for link in cells["links"]
            ],
            dates=cells["dates"],
            places=cells["places"],
            infoboxes=cells["infoboxes"],
            markup=cells["markup"],
            source=cells["source"],
            utf8=utf8,
        )

    def find(self, key: WikiKey | bytes) -> Document | None:
        """The document holding `key`, scanning the key columns of the stems."""
        raw = (key.article if isinstance(key, WikiKey) else WikiKey.decode(key).article).encode()
        for first, _, path in self.stems:
            index = pc.index(
                pq.read_table(path, columns=["key"])["key"], pa.scalar(raw, pa.binary(16))
            ).as_py()
            if index >= 0:
                return self.document(first + index)
        return None

    def vectors(self, model: str, first: int, count: int, part: str = "body") -> np.ndarray:
        """Float16 vectors of documents `[first, first + count)` as a matrix, or of all their sections for `gte`.

        The `gte` rows of one document together form its MaxSim query.
        """
        column = "vectors" if model == "gte" else part
        cells = self._rows(f".{model}.parquet", first, count, [column])[column].combine_chunks()
        values = cells.flatten().flatten() if model == "gte" else cells.flatten()
        return values.to_numpy().reshape(-1, DIMENSIONS[model])

    def graph(self, model: str, residency: str = "ram"):
        """The `.safegraphs` index of a model, loaded into RAM or served from NVMe in place."""
        import usearch

        if (model, residency) not in self._graphs:
            path = self.folder / f"{GRAPHS[model]}.safegraphs"
            loader = (
                usearch.DenseIndex.load_ram if residency == "ram" else usearch.DenseIndex.view_nvme
            )
            executor = (
                usearch.Executor.for_ram() if residency == "ram" else usearch.Executor.for_nvme()
            )
            self._graphs[model, residency] = (loader(path, executor=executor), executor)
        return self._graphs[model, residency]

    def search(
        self,
        model: str,
        queries: np.ndarray,
        count: int,
        expansion: int = 0,
        residency: str = "ram",
    ) -> list[tuple[WikiKey, float]]:
        """Nearest documents to one query: a single vector, or for `gte` all of a document's section vectors scored by MaxSim."""
        import usearch

        index, executor = self.graph(model, residency)
        arena = usearch.Arena(executor)
        queries = np.ascontiguousarray(np.atleast_2d(queries))
        if model == "gte":
            products = index.on(executor).maxsim(
                queries, len(queries), count, arena, expansion_factor=expansion
            )
            keys, scores = products.query(0)
            return [
                (WikiKey.decode(bytes(key)), float(score))
                for key, score in zip(np.asarray(keys), np.asarray(scores), strict=True)
            ]
        matches = index.on(executor).search(queries[:1], count, arena, expansion_factor=expansion)
        return [(WikiKey.decode(match[0].bytes), float(match[1])) for match in matches[0]]

    def search_like(
        self, row: int, model: str, count: int, expansion: int = 0
    ) -> list[tuple[WikiKey, float]]:
        """Documents nearest to document `row` under `model`, which may include the document itself."""
        return self.search(model, self.vectors(model, row, 1), count, expansion)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Print one USearchWiki document with its sections, links, dates and places."
    )
    parser.add_argument("folder", help="a collection folder, such as data/example")
    parser.add_argument("row", type=int, help="document row within the collection")
    arguments = parser.parse_args()
    dataset = WikiDataset.open(arguments.folder)
    document = dataset.document(arguments.row)
    print(f"{document.collection} {document.title!r} {document.url} {document.license}")
    entity = f"Q{document.entity}" if document.entity else "none"
    print(f"entity {entity}, markup {document.markup}, authors {document.authors}")
    for index, section in enumerate(document.sections):
        print(f"\n{'#' * max(section.level, 1)} {section.heading or document.title}")
        for link in document.links:
            if link["section"] == index:
                print(
                    f"  link {document.quote(link['span'])!r} -> {'Q' + str(link['entity']) if link['entity'] else '…'} {link['target'] or link['url']}"
                )
        for date in document.dates:
            if date["section"] == index:
                start, end = civil_from_days(date["start"]), civil_from_days(date["end"])
                print(
                    f"  date {document.quote(date['span'])!r} {start}..{end} via {SOURCES[date['source']]}"
                )
        for place in document.places:
            if place["section"] == index:
                box = place["south"], place["west"], place["north"], place["east"]
                print(
                    f"  place {document.quote(place['span'])!r} {box} via {SOURCES[place['source']]}"
                )
    outside = sum(link["section"] is None for link in document.links)
    print(
        f"\n{outside} links outside sections, "
        f"{len(document.subject_dates)} subject dates, {len(document.subject_places)} subject places"
    )
