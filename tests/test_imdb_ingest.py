"""Exercise the IMDb ingest against miniature dumps.

The real dumps are gigabytes and take an hour to fetch, so the join logic —
which is where the subtle errors live, particularly the language resolution —
is tested against hand-written fixtures with known answers.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from entertainer import config
from entertainer.data import imdb

BASICS = [
    "tconst\ttitleType\tprimaryTitle\toriginalTitle\tisAdult\tstartYear\tendYear\truntimeMinutes\tgenres",
    "tt01\tmovie\tKumbalangi Nights\tKumbalangi Nights\t0\t2019\t\\N\t135\tComedy,Drama",
    "tt02\tmovie\tThe Godfather\tThe Godfather\t0\t1972\t\\N\t175\tCrime,Drama",
    "tt03\ttvSeries\tBreaking Bad\tBreaking Bad\t0\t2008\t2013\t49\tCrime,Drama",
    # Filtered out: adult, too old, wrong type, below every vote floor.
    "tt04\tmovie\tAdult Thing\tAdult Thing\t1\t2010\t\\N\t90\tDrama",
    "tt05\tmovie\tAncient\tAncient\t0\t1910\t\\N\t60\tDrama",
    "tt06\tshort\tA Short\tA Short\t0\t2015\t\\N\t12\tComedy",
    "tt07\tmovie\tObscure\tObscure\t0\t2015\t\\N\t90\tDrama",
    # Kannada film with modest votes: must survive on the lower floor.
    "tt08\tmovie\tKantara\tKantara\t0\t2022\t\\N\t148\tAction,Drama",
]

RATINGS = [
    "tconst\taverageRating\tnumVotes",
    "tt01\t8.4\t40000",
    "tt02\t9.2\t2000000",
    "tt03\t9.5\t2100000",
    "tt04\t6.0\t5000",
    "tt05\t7.0\t9000",
    "tt06\t7.5\t9000",
    "tt07\t7.9\t80",
    "tt08\t8.2\t150",
]

AKAS = [
    "titleId\tordering\ttitle\tregion\tlanguage\ttypes\tattributes\tisOriginalTitle",
    "tt01\t1\tKumbalangi Nights\tIN\tml\t\\N\t\\N\t1",
    "tt01\t2\tKumbalangi Nights\tUS\t\\N\t\\N\t\\N\t0",
    "tt01\t3\tKumbalangi Nights\tIN\tta\t\\N\t\\N\t0",
    "tt01\t4\tKumbalangi Nights\tIN\tta\t\\N\t\\N\t0",
    "tt02\t1\tThe Godfather\tUS\t\\N\t\\N\t\\N\t1",
    "tt02\t2\tLe Parrain\tFR\t\\N\t\\N\t\\N\t0",
    "tt03\t1\tBreaking Bad\tUS\t\\N\t\\N\t\\N\t1",
    "tt08\t1\tKantara\tIN\tkn\t\\N\t\\N\t1",
]

CREW = [
    "tconst\tdirectors\twriters",
    "tt01\tnm01\tnm02",
    "tt02\tnm03\tnm03",
    "tt03\tnm04\tnm04",
    "tt08\tnm05\tnm05",
]

PRINCIPALS = [
    "tconst\tordering\tnconst\tcategory\tjob\tcharacters",
    "tt01\t1\tnm10\tactor\t\\N\t\\N",
    "tt01\t2\tnm11\tactor\t\\N\t\\N",
    "tt01\t99\tnm12\tactor\t\\N\t\\N",
    "tt02\t1\tnm13\tactor\t\\N\t\\N",
    "tt03\t1\tnm14\tactor\t\\N\t\\N",
]

NAMES = [
    "nconst\tprimaryName\tbirthYear\tdeathYear\tprimaryProfession\tknownForTitles",
    "nm01\tMadhu C. Narayanan\t\\N\t\\N\tdirector\t\\N",
    "nm02\tShyam Pushkaran\t\\N\t\\N\twriter\t\\N",
    "nm03\tFrancis Ford Coppola\t\\N\t\\N\tdirector\t\\N",
    "nm04\tVince Gilligan\t\\N\t\\N\tdirector\t\\N",
    "nm05\tRishab Shetty\t\\N\t\\N\tdirector\t\\N",
    "nm10\tShane Nigam\t\\N\t\\N\tactor\t\\N",
    "nm11\tFahadh Faasil\t\\N\t\\N\tactor\t\\N",
    "nm12\tExtra Person\t\\N\t\\N\tactor\t\\N",
    "nm13\tMarlon Brando\t\\N\t\\N\tactor\t\\N",
    "nm14\tBryan Cranston\t\\N\t\\N\tactor\t\\N",
]

FIXTURES = {
    "title.basics.tsv.gz": BASICS,
    "title.ratings.tsv.gz": RATINGS,
    "title.akas.tsv.gz": AKAS,
    "title.crew.tsv.gz": CREW,
    "title.principals.tsv.gz": PRINCIPALS,
    "name.basics.tsv.gz": NAMES,
}


@pytest.fixture()
def mini_dumps(tmp_path: Path, monkeypatch):
    root = tmp_path / "raw" / "imdb"
    root.mkdir(parents=True)
    for name, lines in FIXTURES.items():
        with gzip.open(root / name, "wt", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

    class P:
        raw = tmp_path / "raw"

    monkeypatch.setattr(imdb, "PATHS", P)
    monkeypatch.setattr(config, "PATHS", P, raising=False)
    return root


def test_build_filters_and_joins(mini_dumps):
    df = imdb.build()
    ids = set(df["imdb_id"].to_list())
    assert ids == {"tt01", "tt02", "tt03", "tt08"}, "unexpected survivors"


def test_language_uses_the_original_title_row(mini_dumps):
    df = imdb.build()
    langs = dict(zip(df["imdb_id"].to_list(), df["language"].to_list(), strict=True))
    # Two Tamil akas rows outnumber the single Malayalam one, but the
    # Malayalam row is the original title and must win.
    assert langs["tt01"] == "ml"
    assert langs["tt08"] == "kn"
    # No language tag anywhere, so the region stands in.
    assert langs["tt02"] == "en"


def test_small_industry_floor_keeps_a_150_vote_kannada_film(mini_dumps):
    df = imdb.build()
    row = df.filter(df["imdb_id"] == "tt08")
    assert row.height == 1
    assert row["imdb_votes"][0] == 150
    assert config.VOTE_FLOOR_BY_LANGUAGE["kn"] <= 150 < config.VOTE_FLOOR_DEFAULT


def test_crew_and_cast_resolve_to_names(mini_dumps):
    df = imdb.build()
    row = df.filter(df["imdb_id"] == "tt01").to_dicts()[0]
    assert "Madhu C. Narayanan" in row["directors"]
    assert "Shyam Pushkaran" in row["writers"]
    assert "Shane Nigam" in row["cast_names"]
    # ordering 99 is beyond the top-cast cut.
    assert "Extra Person" not in row["cast_names"]


def test_series_are_tagged_as_tv(mini_dumps):
    df = imdb.build()
    kinds = dict(zip(df["imdb_id"].to_list(), df["kind"].to_list(), strict=True))
    assert kinds["tt03"] == "tv"
    assert kinds["tt02"] == "movie"


def test_degrades_when_optional_dumps_are_absent(mini_dumps):
    (mini_dumps / "title.akas.tsv.gz").unlink()
    (mini_dumps / "title.principals.tsv.gz").unlink()
    df = imdb.build()
    assert df.height >= 1
    # Without akas every language is unknown, so the default floor applies
    # and only the heavily-voted titles survive.
    assert set(df["language"].to_list()) == {"xx"}
    assert set(df["imdb_id"].to_list()) == {"tt01", "tt02", "tt03"}
