"""Keys and day counts as the published shards encode them, and one local collection read back."""

import datetime
import os
from pathlib import Path

import pytest

from usearchwiki import WikiDataset, WikiKey, civil_from_days

EXAMPLE = Path(os.environ.get("USEARCHWIKI_EXAMPLE", Path(__file__).parents[1] / "data/example"))


def test_key_round_trip_keeps_uuid_version_and_variant():
    key = WikiKey(collection=74, page=9646932, revision=1295127271, section=3)
    raw = key.encode()
    assert raw[6] >> 4 == 8 and raw[8] >> 6 == 2
    assert WikiKey.decode(raw) == key
    assert WikiKey.decode(raw).article == WikiKey(74, 9646932, 1295127271)


def test_entity_keys_carry_the_qid():
    assert WikiKey(0xFFFF, 90, 0).qid == 90
    assert WikiKey(74, 90, 1).qid is None


def test_days_match_the_calendar_and_reach_before_year_one():
    for date in [
        datetime.date(1970, 1, 1),
        datetime.date(1789, 7, 14),
        datetime.date(1, 1, 1),
        datetime.date(2026, 11, 3),
    ]:
        days = (date - datetime.date(1970, 1, 1)).days
        assert civil_from_days(days) == (date.year, date.month, date.day)
    assert civil_from_days((datetime.date(1, 1, 1) - datetime.date(1970, 1, 1)).days - 1) == (
        0,
        12,
        31,
    )


@pytest.mark.skipif(
    not (EXAMPLE.parent / "collections.parquet").exists(), reason="no local example collection"
)
def test_example_sections_tile_the_text_and_carry_one_gte_vector_each():
    dataset = WikiDataset.open(EXAMPLE)
    for row in [0, 4095, 4096, len(dataset) - 1]:
        document = dataset.document(row)
        assert dataset.collections[document.key.collection] == document.collection
        assert "".join(section.text for section in document.sections) == document.text
        assert len(dataset.vectors("gte", row, 1)) == len(document.sections)
    assert dataset.vectors("qwen", 4095, 2).shape == (2, 1024)
    assert dataset.find(document.key).row == len(dataset) - 1
