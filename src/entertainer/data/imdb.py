"""Build the base catalogue from IMDb's bulk TSV exports.

IMDb is the spine of the catalogue rather than TMDB because it is complete,
free, unmetered, and — critically for this project — it carries small-industry
cinema (Malayalam, Kannada, Persian) with the same fidelity as Hollywood.
TMDB then enriches on top of it; see ``tmdb.py``.

The whole pipeline runs lazily through Polars' streaming engine, so the 11M
title / 95M principal rows never need to fit in memory at once.
"""

from __future__ import annotations

import polars as pl
from rich.console import Console

from ..config import KEPT_TITLE_TYPES, MIN_YEAR, PATHS

console = Console()

_READ = dict(separator="\t", quote_char=None, null_values=["\\N"], infer_schema_length=0)

def _raw(name: str):
    path = PATHS.raw / "imdb" / name
    if not path.exists():
        raise FileNotFoundError(f"missing IMDb dump {path}; run `entertainer fetch` first")
    return pl.scan_csv(path, **_READ)


def available(name: str) -> bool:
    """Whether an optional dump has finished downloading.

    ``title.basics`` and ``title.ratings`` are required. The rest — languages,
    crew, cast — enrich the item card but the catalogue is still usable
    without them, which matters because they are also by far the largest
    files. A partial build is better than no build.
    """
    return (PATHS.raw / "imdb" / name).exists()


def _basics() -> pl.LazyFrame:
    return (
        _raw("title.basics.tsv.gz")
        .select(
            "tconst", "titleType", "primaryTitle", "originalTitle",
            "isAdult", "startYear", "endYear", "runtimeMinutes", "genres",
        )
        .filter(pl.col("titleType").is_in(list(KEPT_TITLE_TYPES)))
        .with_columns(
            year=pl.col("startYear").cast(pl.Int32, strict=False),
            end_year=pl.col("endYear").cast(pl.Int32, strict=False),
            runtime=pl.col("runtimeMinutes").cast(pl.Int32, strict=False),
            adult=pl.col("isAdult").cast(pl.Int8, strict=False).fill_null(0) == 1,
        )
        .filter(pl.col("year") >= MIN_YEAR, ~pl.col("adult"))
        .with_columns(
            kind=pl.when(pl.col("titleType").str.starts_with("tv") & (pl.col("titleType") != "tvMovie"))
            .then(pl.lit("tv"))
            .otherwise(pl.lit("movie")),
            genres=pl.col("genres").fill_null("").str.split(","),
        )
        .select(
            "tconst", "kind", "year", "end_year", "runtime", "genres", "adult",
            pl.col("primaryTitle").alias("title"),
            pl.col("originalTitle").alias("original_title"),
        )
    )


def _ratings() -> pl.LazyFrame:
    return _raw("title.ratings.tsv.gz").select(
        "tconst",
        pl.col("averageRating").cast(pl.Float64, strict=False).alias("imdb_rating"),
        pl.col("numVotes").cast(pl.Int32, strict=False).alias("imdb_votes"),
    )


def _countries(keep: pl.LazyFrame) -> pl.LazyFrame:
    """Release regions per title, from akas.

    Note what this deliberately does *not* do: derive a production language.

    An earlier version of this pipeline did, and it was badly wrong. IMDb's
    akas ``language`` column records the language of a *localised release*,
    not of the film. Kantara — a Kannada film — carries tags for English,
    French, Hindi, Japanese and Turkish, and none for Kannada. Baahubali
    carries Tamil, Telugu, Hindi and English with no indication which is the
    original. The ``isOriginalTitle`` row, which ought to settle it, has a
    null language on essentially every title checked.

    Worse, the failure was silent and biased in one direction: every
    non-English film with a US or UK release picked up an ``en`` tag, so the
    catalogue came out 79% English with zero Tamil, Malayalam or Telugu
    titles — while looking entirely plausible.

    Production language now comes from TMDB's ``original_language``, which is
    authoritative, and the per-language vote floors are applied after
    enrichment rather than during the build. See ``catalog.prune_by_language``.
    """
    return (
        _raw("title.akas.tsv.gz")
        .select(pl.col("titleId").alias("tconst"), "region")
        .join(keep.select("tconst"), on="tconst", how="semi")
        .filter(pl.col("region").is_not_null())
        .group_by("tconst")
        .agg(pl.col("region").unique().head(12).alias("countries"))
    )


def _names() -> pl.LazyFrame:
    return _raw("name.basics.tsv.gz").select("nconst", pl.col("primaryName").alias("name"))


def _crew(keep: pl.LazyFrame) -> pl.LazyFrame:
    names = _names()
    crew = (
        _raw("title.crew.tsv.gz")
        .select("tconst", "directors", "writers")
        .join(keep.select("tconst"), on="tconst", how="semi")
    )

    def resolve(col: str, out: str) -> pl.LazyFrame:
        return (
            crew.select("tconst", pl.col(col).str.split(",").alias("nconst"))
            .explode("nconst", empty_as_null=True)
            .filter(pl.col("nconst").is_not_null())
            .join(names, on="nconst", how="inner")
            .group_by("tconst")
            .agg(pl.col("name").unique().head(4).alias(out))
        )

    return resolve("directors", "directors").join(
        resolve("writers", "writers"), on="tconst", how="full", coalesce=True
    )


def _cast(keep: pl.LazyFrame, top_n: int = 8) -> pl.LazyFrame:
    return (
        _raw("title.principals.tsv.gz")
        .select(
            "tconst",
            pl.col("ordering").cast(pl.Int16, strict=False),
            "nconst",
            "category",
        )
        .filter(
            pl.col("category").is_in(["actor", "actress", "self"]),
            pl.col("ordering") <= top_n,
        )
        .join(keep.select("tconst"), on="tconst", how="semi")
        .join(_names(), on="nconst", how="inner")
        .sort("ordering")
        .group_by("tconst")
        .agg(pl.col("name").head(top_n).alias("cast_names"))
    )


def build(min_votes: int = 50) -> pl.DataFrame:
    """Produce the filtered, joined IMDb catalogue as an eager DataFrame.

    The floor here is flat and deliberately low. Language-aware pruning — the
    thing that stops Hollywood-calibrated thresholds from erasing small
    industries — happens after TMDB enrichment, because that is the first
    point at which the production language is actually known. Filtering on a
    language you have guessed wrong is worse than not filtering at all.
    """
    base = _basics().join(_ratings(), on="tconst", how="left")
    joined = base.filter(pl.col("imdb_votes") >= min_votes)

    if available("title.akas.tsv.gz"):
        console.print("[dim]resolving release regions from title.akas (50M rows)…[/dim]")
        joined = joined.join(_countries(joined), on="tconst", how="left")
    else:
        console.print("[yellow]title.akas missing — no release regions[/yellow]")
        joined = joined.with_columns(countries=pl.lit(None, dtype=pl.List(pl.Utf8)))

    if available("title.crew.tsv.gz") and available("name.basics.tsv.gz"):
        console.print("[dim]resolving crew from title.crew…[/dim]")
        joined = joined.join(_crew(joined), on="tconst", how="left")
    else:
        console.print("[yellow]title.crew missing — no director or writer credits[/yellow]")
        joined = joined.with_columns(
            directors=pl.lit(None, dtype=pl.List(pl.Utf8)),
            writers=pl.lit(None, dtype=pl.List(pl.Utf8)),
        )

    if available("title.principals.tsv.gz") and available("name.basics.tsv.gz"):
        console.print("[dim]resolving cast from title.principals (95M rows)…[/dim]")
        joined = joined.join(_cast(joined), on="tconst", how="left")
    else:
        console.print("[yellow]title.principals missing — no cast[/yellow]")
        joined = joined.with_columns(cast_names=pl.lit(None, dtype=pl.List(pl.Utf8)))

    df = joined.collect(engine="streaming")
    return df.with_columns(
        imdb_id=pl.col("tconst"),
        # Unknown until TMDB says otherwise.
        language=pl.lit("xx", dtype=pl.Utf8),
        languages=pl.lit(None).cast(pl.List(pl.Utf8)),
        countries=pl.col("countries").fill_null([]),
        directors=pl.col("directors").fill_null([]),
        writers=pl.col("writers").fill_null([]),
        cast_names=pl.col("cast_names").fill_null([]),
    ).drop("tconst")
