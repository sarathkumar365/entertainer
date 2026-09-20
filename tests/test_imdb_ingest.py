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


def test_build_filters_out_shorts_adult_and_pre_1930(mini_dumps):
    df = imdb.build(min_votes=100)
    ids = set(df["imdb_id"].to_list())
    # tt04 adult, tt05 pre-1930, tt06 a short, tt07 below the flat floor.
    assert ids == {"tt01", "tt02", "tt03", "tt08"}, ids


def test_build_does_not_guess_a_language(mini_dumps):
    """akas tags describe localised releases, not the production language.

    tt01 is a Malayalam film whose akas rows are mostly Tamil dubs and a US
    release. Any attempt to infer a language from that is wrong, so the build
    declines to try and leaves it for TMDB.
    """
    df = imdb.build(min_votes=100)
    assert set(df["language"].to_list()) == {"xx"}


def test_low_vote_titles_survive_the_flat_floor_for_later_pruning(mini_dumps):
    df = imdb.build(min_votes=50)
    ids = set(df["imdb_id"].to_list())
    # The 150-vote Kannada film must reach the enrichment stage; whether it
    # survives is decided afterwards, by its real language's floor.
    assert "tt08" in ids
    assert config.VOTE_FLOOR_BY_LANGUAGE["kn"] <= 150 < config.VOTE_FLOOR_DEFAULT


def test_release_regions_are_collected(mini_dumps):
    df = imdb.build(min_votes=100)
    row = df.filter(df["imdb_id"] == "tt02").to_dicts()[0]
    assert set(row["countries"]) == {"US", "FR"}


def test_crew_and_cast_resolve_to_names(mini_dumps):
    df = imdb.build(min_votes=100)
    row = df.filter(df["imdb_id"] == "tt01").to_dicts()[0]
    assert "Madhu C. Narayanan" in row["directors"]
    assert "Shyam Pushkaran" in row["writers"]
    assert "Shane Nigam" in row["cast_names"]
    # ordering 99 is beyond the top-cast cut.
    assert "Extra Person" not in row["cast_names"]


def test_series_are_tagged_as_tv(mini_dumps):
    df = imdb.build(min_votes=100)
    kinds = dict(zip(df["imdb_id"].to_list(), df["kind"].to_list(), strict=True))
    assert kinds["tt03"] == "tv"
    assert kinds["tt02"] == "movie"


def test_degrades_when_optional_dumps_are_absent(mini_dumps):
    (mini_dumps / "title.akas.tsv.gz").unlink()
    (mini_dumps / "title.principals.tsv.gz").unlink()
    df = imdb.build(min_votes=100)
    assert set(df["imdb_id"].to_list()) == {"tt01", "tt02", "tt03", "tt08"}
    assert all(not c for c in df["countries"].to_list())
    assert all(not c for c in df["cast_names"].to_list())
    # Crew is still present: it comes from a different dump.
    assert any(d for d in df["directors"].to_list())


def test_rebuild_preserves_enrichment(mini_dumps, tmp_path, monkeypatch):
    """A rebuild must never discard hundreds of thousands of network round trips.

    Enrichment is by far the most expensive stage in the pipeline, and a
    rebuild is routine — it is how MovieLens identities and cast credits get
    picked up once those downloads finish. Carrying enrichment across is keyed
    on the IMDb id rather than the internal item_id, because item ids are
    positional within a build and are reassigned every time.
    """
    from entertainer import store
    from entertainer.data import catalog

    class P:
        raw = tmp_path / "raw"
        root = tmp_path
        interim = tmp_path / "interim"
        embeddings = tmp_path / "emb"
        artifacts = tmp_path / "art"
        reports = tmp_path / "rep"
        catalog_db = tmp_path / "test.duckdb"

        @classmethod
        def ensure(cls):
            for p in (cls.raw, cls.interim, cls.embeddings, cls.artifacts, cls.reports):
                p.mkdir(parents=True, exist_ok=True)
            return cls

    monkeypatch.setattr(store, "PATHS", P)
    monkeypatch.setattr(catalog, "PATHS", P)

    assert catalog.build_base(min_votes=100) == 4

    con = store.connect()
    con.execute(
        """
        UPDATE titles SET language = 'ml', overview = 'Four brothers.',
                          keywords = ['family'], tmdb_id = 12345, enriched_at = now()
        WHERE imdb_id = 'tt01'
        """
    )
    # Shift every item_id so a carry keyed on position would visibly fail.
    con.execute("UPDATE titles SET item_id = item_id + 900")
    con.close()

    assert catalog.build_base(min_votes=100) == 4

    con = store.connect(read_only=True)
    row = con.execute(
        "SELECT language, overview, keywords, tmdb_id, enriched_at FROM titles WHERE imdb_id = 'tt01'"
    ).fetchone()
    others = con.execute(
        "SELECT count(*) FROM titles WHERE enriched_at IS NOT NULL"
    ).fetchone()[0]
    con.close()

    assert row[0] == "ml", "TMDB language must survive the rebuild"
    assert row[1] == "Four brothers."
    assert list(row[2]) == ["family"]
    assert row[3] == 12345
    assert row[4] is not None
    assert others == 1, "only the enriched row should be marked enriched"


def test_rebuild_from_empty_carries_nothing_and_does_not_fail(mini_dumps, tmp_path, monkeypatch):
    from entertainer import store
    from entertainer.data import catalog

    class P:
        raw = tmp_path / "raw"
        root = tmp_path
        interim = tmp_path / "i"
        embeddings = tmp_path / "e"
        artifacts = tmp_path / "a"
        reports = tmp_path / "r"
        catalog_db = tmp_path / "fresh.duckdb"

        @classmethod
        def ensure(cls):
            for p in (cls.raw, cls.interim, cls.embeddings, cls.artifacts, cls.reports):
                p.mkdir(parents=True, exist_ok=True)
            return cls

    monkeypatch.setattr(store, "PATHS", P)
    monkeypatch.setattr(catalog, "PATHS", P)
    assert catalog.build_base(min_votes=100) == 4
    assert catalog.build_base(min_votes=100) == 4
