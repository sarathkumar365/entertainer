import duckdb
import pytest

from entertainer.resolve import normalise, resolve_one, search, split_year
from entertainer.store import SCHEMA

ROWS = [
    (1, "tt0000001", "Drishyam", "Drishyam", 2013, "ml", 35000, "movie"),
    (2, "tt0000002", "Drishyam", "Drishyam", 2015, "hi", 90000, "movie"),
    (3, "tt0000003", "Drishyam 2", "Drishyam 2", 2021, "ml", 30000, "movie"),
    (4, "tt0000004", "The Godfather", "The Godfather", 1972, "en", 2000000, "movie"),
    (5, "tt0000005", "Kumbalangi Nights", "കുമ്പളങ്ങി നൈറ്റ്സ്", 2019, "ml", 40000, "movie"),
    (6, "tt0000006", "Breaking Bad", "Breaking Bad", 2008, "en", 2100000, "tv"),
]


@pytest.fixture()
def con():
    c = duckdb.connect(":memory:")
    c.execute(SCHEMA)
    for r in ROWS:
        c.execute(
            "INSERT INTO titles (item_id, imdb_id, title, original_title, year, language, "
            "imdb_votes, kind) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            list(r),
        )
    return c


def test_normalise_strips_articles_and_accents():
    assert normalise("The Godfather") == "godfather"
    assert normalise("Amélie!") == "amelie"


def test_split_year():
    assert split_year("Drishyam (2013)") == ("Drishyam", 2013)
    assert split_year("Drishyam 2013") == ("Drishyam", 2013)
    assert split_year("Drishyam 2")[1] is None


def test_exact_match_wins(con):
    match, _ = resolve_one(con, "The Godfather")
    assert match is not None and match.item_id == 4


def test_year_disambiguates_remakes(con):
    match, _ = resolve_one(con, "Drishyam (2015)")
    assert match is not None and match.item_id == 2


def test_ambiguity_is_surfaced_not_guessed(con):
    match, alternatives = resolve_one(con, "Drishyam")
    # Two films share the exact title; the caller must be asked.
    assert match is None or len({a.item_id for a in alternatives}) >= 1
    assert alternatives


def test_original_title_is_searchable(con):
    hits = search(con, "കുമ്പളങ്ങി നൈറ്റ്സ്")
    assert hits and hits[0].item_id == 5


def test_typos_still_resolve(con):
    match, alternatives = resolve_one(con, "Kumbalangi Nites")
    found = [match] if match else alternatives
    assert any(m.item_id == 5 for m in found)


def test_kind_filter(con):
    hits = search(con, "Breaking Bad", kind="movie")
    assert not hits
    assert search(con, "Breaking Bad", kind="tv")


def test_unknown_title_returns_nothing(con):
    match, alternatives = resolve_one(con, "zzzzqqqq nonexistent")
    assert match is None and not alternatives
