"""Assemble the unified catalogue in DuckDB.

Order of operations: IMDb bulk gives the spine, MovieLens links attach the
collaborative-filtering identity, TMDB fills in prose and language. Each stage
is idempotent so the pipeline can be re-run or resumed without redoing work.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import polars as pl
from rich.console import Console

from ..config import PATHS, VOTE_FLOOR_BY_LANGUAGE, VOTE_FLOOR_DEFAULT
from ..store import connect, set_meta
from . import imdb as imdb_mod

console = Console()

_COLUMNS = [
    "item_id", "imdb_id", "tmdb_id", "movielens_id", "kind", "title", "original_title",
    "year", "end_year", "runtime", "genres", "language", "languages", "countries",
    "imdb_rating", "imdb_votes", "tmdb_rating", "tmdb_votes", "popularity",
    "directors", "writers", "cast_names", "keywords", "overview", "tagline",
    "poster_path", "adult", "quality", "enriched_at",
]


def _movielens_links() -> pl.DataFrame:
    path = PATHS.raw / "ml-32m" / "links.csv"
    if not path.exists():
        console.print("[yellow]MovieLens links.csv missing; skipping CF identity join[/yellow]")
        return pl.DataFrame({"imdb_id": [], "movielens_id": [], "tmdb_id_ml": []},
                            schema={"imdb_id": pl.Utf8, "movielens_id": pl.Int32,
                                    "tmdb_id_ml": pl.Int32})
    return (
        pl.read_csv(path, schema_overrides={"imdbId": pl.Utf8, "tmdbId": pl.Utf8})
        .select(
            imdb_id="tt" + pl.col("imdbId").str.zfill(7),
            movielens_id=pl.col("movieId").cast(pl.Int32),
            tmdb_id_ml=pl.col("tmdbId").cast(pl.Int32, strict=False),
        )
    )


def quality_prior(rating: float | None, votes: int | None, prior_mean: float = 6.4,
                  prior_weight: float = 2500.0) -> float:
    """Bayesian-shrunk quality on 0..1.

    A 9.2 from 40 voters is not evidence of a better film than an 8.1 from
    400,000. Shrinking towards the global mean stops the recommender from
    surfacing obscure titles purely because a handful of enthusiasts rated
    them highly, without hard-cutting the long tail the way a vote threshold
    would.
    """
    if rating is None or votes is None or votes <= 0:
        return prior_mean / 10.0
    shrunk = (rating * votes + prior_mean * prior_weight) / (votes + prior_weight)
    # A mild log-vote bonus keeps genuinely canonical films above merely
    # inoffensive ones that sit near the prior.
    confidence = math.log1p(votes) / math.log1p(1_000_000)
    return max(0.0, min(1.0, 0.85 * shrunk / 10.0 + 0.15 * confidence))


def build_base(min_votes: int = 50) -> int:
    """Stage 1: IMDb + MovieLens ids into `titles`. Returns row count."""
    df = imdb_mod.build(min_votes=min_votes)
    console.print(f"[green]IMDb catalogue:[/green] {df.height:,} titles")

    df = df.join(_movielens_links(), on="imdb_id", how="left").with_columns(
        movielens_id=pl.col("movielens_id").cast(pl.Int32),
        tmdb_id=pl.col("tmdb_id_ml").cast(pl.Int32),
    ).drop("tmdb_id_ml")

    df = df.sort(["imdb_votes"], descending=True, nulls_last=True).with_row_index("item_id")
    df = df.with_columns(
        item_id=pl.col("item_id").cast(pl.Int32),
        quality=pl.struct(["imdb_rating", "imdb_votes"]).map_elements(
            lambda s: quality_prior(s["imdb_rating"], s["imdb_votes"]), return_dtype=pl.Float64
        ),
        tmdb_rating=pl.lit(None, dtype=pl.Float64),
        tmdb_votes=pl.lit(None, dtype=pl.Int32),
        popularity=pl.lit(None, dtype=pl.Float64),
        keywords=pl.lit(None).cast(pl.List(pl.Utf8)),
        overview=pl.lit(None, dtype=pl.Utf8),
        tagline=pl.lit(None, dtype=pl.Utf8),
        poster_path=pl.lit(None, dtype=pl.Utf8),
        enriched_at=pl.lit(None, dtype=pl.Datetime("us")),
    ).select(_COLUMNS)

    con = connect()
    con.register("_df", df.to_arrow())

    # Item ids are positional within a build, so a rebuild renumbers the whole
    # catalogue. The event log references them. Without remapping, every
    # verdict the user has ever given silently reattaches to a different film
    # — the worst possible failure for a system whose only real asset is that
    # log. The mapping goes through the IMDb id, which is stable.
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE _old_ids AS
        SELECT item_id AS old_id, imdb_id FROM titles WHERE imdb_id IS NOT NULL
        """
    )

    # Enrichment is the most expensive thing in the pipeline by an order of
    # magnitude — hundreds of thousands of network round trips. A rebuild
    # must never discard it. Anything already fetched for an IMDb id is
    # carried across, keyed on that id rather than on the internal item_id,
    # which is positional and changes whenever the catalogue is rebuilt.
    carried = con.execute(
        """
        CREATE OR REPLACE TEMP TABLE _carry AS
        SELECT imdb_id, tmdb_id, overview, tagline, language, popularity,
               tmdb_rating, tmdb_votes, poster_path, keywords, genres,
               enriched_at, keywords_at
        FROM titles WHERE enriched_at IS NOT NULL
        """
    ) and con.execute("SELECT count(*) FROM _carry").fetchone()[0]

    con.execute("DELETE FROM titles")
    # Name the columns explicitly. A positional INSERT breaks the moment a
    # migration adds a column, and it breaks by silently mapping values into
    # the wrong fields if the counts happen to match.
    cols = ", ".join(_COLUMNS)
    con.execute(f"INSERT INTO titles ({cols}) SELECT {cols} FROM _df")
    con.unregister("_df")

    # Remap the event log onto the new numbering before anything reads it.
    remapped = con.execute(
        """
        CREATE OR REPLACE TEMP TABLE _remap AS
        SELECT o.old_id, t.item_id AS new_id
        FROM _old_ids o JOIN titles t USING (imdb_id)
        WHERE o.old_id <> t.item_id
        """
    ) and con.execute("SELECT count(*) FROM _remap").fetchone()[0]

    orphaned = con.execute(
        """
        SELECT count(DISTINCT e.item_id) FROM events e
        LEFT JOIN _old_ids o ON o.old_id = e.item_id
        LEFT JOIN titles t ON t.imdb_id = o.imdb_id
        WHERE t.item_id IS NULL
        """
    ).fetchone()[0]

    if remapped:
        for table in ("events", "impressions"):
            con.execute(
                f"UPDATE {table} SET item_id = r.new_id FROM _remap r "
                f"WHERE {table}.item_id = r.old_id"
            )
        console.print(f"[dim]remapped the event log across {remapped:,} renumbered titles[/dim]")
    if orphaned:
        console.print(
            f"[yellow]{orphaned} rated title(s) are no longer in the catalogue; "
            f"their verdicts are preserved but inactive[/yellow]"
        )

    if carried:
        con.execute(
            """
            UPDATE titles t SET
                tmdb_id     = coalesce(c.tmdb_id, t.tmdb_id),
                overview    = coalesce(c.overview, t.overview),
                tagline     = coalesce(c.tagline, t.tagline),
                language    = coalesce(c.language, t.language),
                popularity  = coalesce(c.popularity, t.popularity),
                tmdb_rating = coalesce(c.tmdb_rating, t.tmdb_rating),
                tmdb_votes  = coalesce(c.tmdb_votes, t.tmdb_votes),
                poster_path = coalesce(c.poster_path, t.poster_path),
                keywords    = coalesce(c.keywords, t.keywords),
                keywords_at = c.keywords_at,
                genres      = list_distinct(list_concat(coalesce(t.genres, []),
                                                        coalesce(c.genres, []))),
                enriched_at = c.enriched_at
            FROM _carry c WHERE t.imdb_id = c.imdb_id
            """
        )
        console.print(f"[green]carried {carried:,} enriched records across the rebuild[/green]")

    set_meta(con, "catalog.min_votes", min_votes)
    set_meta(con, "catalog.size", df.height)
    con.close()
    return df.height


def shrink_rating(
    rating: float | None, votes: int | None, prior_mean: float, prior_weight: float = 2500.0
) -> float | None:
    """Bayesian-shrunk rating on the raw 0..10 scale.

    A 9.2 from 40 voters is not evidence of a better film than an 8.1 from
    400,000. Shrinking towards the mean stops the recommender from surfacing
    obscure titles purely because a handful of enthusiasts rated them highly,
    without hard-cutting the long tail the way a vote threshold would.
    """
    if rating is None or votes is None or votes <= 0:
        return None
    return (rating * votes + prior_mean * prior_weight) / (votes + prior_weight)


def recalibrate_quality(min_titles: int = 200) -> int:
    """Express quality as standing within a title's own industry.

    IMDb rating distributions are not comparable across industries. A film at
    8.1 within Malayalam cinema and one at 8.1 within Hollywood are not making
    the same claim: the two voting populations differ in size, composition and
    enthusiasm. Left absolute, the quality feature becomes a partial proxy for
    language, and the preference model cannot separate the two — someone who
    likes Malayalam films would appear, to the model, to like highly-rated
    ones, and someone who likes highly-rated films would appear to like
    Malayalam.

    Shrinking towards a per-language mean is not enough on its own, which took
    a measurement to notice: shrinkage only pulls *low-vote* titles towards the
    mean, so a well-voted film keeps the offset intact. The feature is
    therefore standardised — how many standard deviations above its own
    industry's norm — which is the only cross-industry comparison that means
    anything. Absolute acclaim is not lost to the model; it still has the raw
    vote count as a separate feature.

    Languages with too few titles to estimate a distribution keep the global
    one, since a mean computed from thirty films is worse than no adjustment.

    Runs after enrichment, because it needs the language.
    """
    con = connect()
    stats = con.execute(
        """
        SELECT language, count(*) AS n, avg(imdb_rating) AS mean, stddev_samp(imdb_rating) AS sd
        FROM titles
        WHERE imdb_rating IS NOT NULL AND imdb_votes > 0
        GROUP BY 1
        """
    ).fetchall()
    g_mean, g_sd = con.execute(
        """
        SELECT avg(imdb_rating), stddev_samp(imdb_rating)
        FROM titles WHERE imdb_rating IS NOT NULL AND imdb_votes > 0
        """
    ).fetchone()
    g_mean = float(g_mean) if g_mean is not None else 6.4
    g_sd = float(g_sd) if g_sd else 1.0

    dist = {}
    for lang, n, mean, sd in stats:
        usable = n >= min_titles and mean is not None and sd and sd > 0.1
        dist[lang] = (float(mean), float(sd)) if usable else (g_mean, g_sd)

    rows = con.execute(
        "SELECT item_id, language, imdb_rating, imdb_votes, tmdb_rating, tmdb_votes FROM titles"
    ).fetchall()
    updates = []
    for item_id, lang, rating, votes, tmdb_rating, tmdb_votes in rows:
        mean, sd = dist.get(lang, (g_mean, g_sd))
        shrunk = shrink_rating(rating, votes, prior_mean=mean)
        if shrunk is None:
            # No IMDb rating — true of anything added by hand through
            # `ent add`. TMDB's rating is on the same 0-10 scale from a
            # smaller pool, so it stands in with a weaker prior weight rather
            # than throwing the information away and calling the title
            # average.
            shrunk = shrink_rating(tmdb_rating, tmdb_votes, prior_mean=mean, prior_weight=400.0)
        if shrunk is None:
            updates.append((0.5, item_id))
            continue
        z = (shrunk - mean) / max(sd, 1e-6)
        standing = 0.5 + 0.18 * z
        # A mild log-vote term keeps genuinely canonical films above merely
        # inoffensive ones sitting at their industry's median.
        confidence = math.log1p(votes or tmdb_votes or 0) / math.log1p(1_000_000)
        updates.append((max(0.0, min(1.0, 0.85 * standing + 0.15 * confidence)), item_id))

    # One set-based update through a temp table, not 270,000 single-row
    # statements. DuckDB is columnar: a row-at-a-time UPDATE rewrites far more
    # than the row, and the difference here is minutes versus seconds.
    frame = pl.DataFrame(
        {
            "item_id": pl.Series([u[1] for u in updates], dtype=pl.Int32),
            "quality": pl.Series([u[0] for u in updates], dtype=pl.Float64),
        }
    )
    con.register("_quality", frame.to_arrow())
    con.execute("CREATE OR REPLACE TEMP TABLE _q AS SELECT * FROM _quality")
    con.unregister("_quality")
    con.execute("UPDATE titles SET quality = q.quality FROM _q q WHERE titles.item_id = q.item_id")
    set_meta(con, "catalog.quality_calibration", "per-language-standardised")
    con.close()
    console.print(
        f"[dim]quality standardised within {len(dist)} languages "
        f"(global mean {g_mean:.2f}, sd {g_sd:.2f})[/dim]"
    )
    return len(updates)


def prune_by_language(scale: float = 1.0, dry_run: bool = False) -> tuple[int, int]:
    """Apply the per-language vote floors, now that the language is known.

    This runs after TMDB enrichment rather than during the build, because
    TMDB's ``original_language`` is the only trustworthy source of a
    production language — IMDb's akas tags describe localised releases, not
    the film. See the note in ``imdb._countries``.

    The floors themselves are the point of the exercise: 200 votes on a
    Malayalam film represents roughly the cultural footprint of 2,000 on an
    English one, and a single global threshold silently deletes whole
    industries. Titles TMDB could not identify keep the default floor, which
    is strict — an unidentifiable title with few votes is usually noise.
    """
    con = connect()
    floors = [(lang, int(round(floor * scale)))
              for lang, floor in VOTE_FLOOR_BY_LANGUAGE.items()]
    con.execute("CREATE OR REPLACE TEMP TABLE _floors (language VARCHAR, floor INTEGER)")
    con.executemany("INSERT INTO _floors VALUES (?, ?)", floors)

    default_floor = int(round(VOTE_FLOOR_DEFAULT * scale))
    # Never prune a title the user has an opinion about. The vote floor is a
    # judgement about what is worth *offering*; it has no business deleting
    # something the person has already told the engine they watched.
    # Three exemptions, each for a different reason:
    #   - a title the user has an opinion about is not the floor's business;
    #   - a title with no IMDb vote count was never measured on this scale, so
    #     applying a threshold to it is meaningless rather than strict — this
    #     is how titles added by hand through `ent add` survive;
    #   - the floor itself only applies where we know the language.
    condition = f"""
        t.imdb_votes IS NOT NULL
        AND t.imdb_votes <
            coalesce((SELECT floor FROM _floors f WHERE f.language = t.language),
                     {default_floor})
        AND t.item_id NOT IN (SELECT DISTINCT item_id FROM events)
    """
    before = con.execute("SELECT count(*) FROM titles").fetchone()[0]
    doomed = con.execute(f"SELECT count(*) FROM titles t WHERE {condition}").fetchone()[0]
    if not dry_run:
        con.execute(f"DELETE FROM titles t WHERE {condition}")
        set_meta(con, "catalog.language_floor_scale", scale)
        set_meta(con, "catalog.size", before - doomed)
    con.close()
    return before, before - doomed


def pending_enrichment(limit: int | None = None) -> list[str]:
    con = connect(read_only=True)
    sql = "SELECT imdb_id FROM titles WHERE enriched_at IS NULL ORDER BY imdb_votes DESC NULLS LAST"
    if limit:
        sql += f" LIMIT {int(limit)}"
    rows = [r[0] for r in con.execute(sql).fetchall()]
    con.close()
    return rows


def apply_enrichment(con, results: Sequence) -> None:
    """Write a batch of TMDB results back into `titles`.

    Fields are merged rather than overwritten: TMDB's language and prose win
    (they are authoritative), but IMDb's rating and vote count stay, because
    IMDb's ratings pool is far larger and better calibrated.
    """
    payload = [
        (
            r.tmdb_id, r.overview, r.tagline, r.original_language, r.popularity,
            r.tmdb_rating, r.tmdb_votes, r.poster_path,
            r.keywords or [], r.keywords or [],
            r.genres or [], r.imdb_id,
        )
        for r in results
    ]
    con.executemany(
        """
        UPDATE titles SET
            tmdb_id     = coalesce(?, tmdb_id),
            overview    = coalesce(?, overview),
            tagline     = coalesce(?, tagline),
            language    = coalesce(?, language),
            popularity  = coalesce(?, popularity),
            tmdb_rating = coalesce(?, tmdb_rating),
            tmdb_votes  = coalesce(?, tmdb_votes),
            poster_path = coalesce(?, poster_path),
            -- Never overwrite fetched keywords with an empty list: the
            -- enrichment pass runs with keywords disabled, and clobbering
            -- them here would silently undo the keyword backfill.
            keywords    = CASE WHEN len(?) > 0 THEN ? ELSE keywords END,
            genres      = list_distinct(list_concat(coalesce(genres, []), ?)),
            enriched_at = now()
        WHERE imdb_id = ?
        """,
        payload,
    )


def language_histogram(top: int = 25) -> list[tuple[str, int]]:
    con = connect(read_only=True)
    rows = con.execute(
        # The language tiebreaker is load-bearing, not cosmetic: without it two
        # languages on the same count come back in whatever order the group-by
        # produced, so the same catalogue prints a different table run to run.
        "SELECT language, count(*) c FROM titles GROUP BY 1 "
        "ORDER BY c DESC, language ASC LIMIT ?",
        [top],
    ).fetchall()
    con.close()
    return rows


def insert_title(con, row: dict) -> int:
    """Insert a single title fetched directly from TMDB. Returns its item_id.

    Item ids are positional within a build, so a title added afterwards takes
    the next free id above everything the build produced. That id survives
    until the next rebuild, at which point it is reassigned — which is exactly
    why the event log joins to `imdb_id` on export and why enrichment is
    carried across on `imdb_id` rather than `item_id`.
    """
    existing = None
    if row.get("imdb_id"):
        existing = con.execute(
            "SELECT item_id FROM titles WHERE imdb_id = ?", [row["imdb_id"]]
        ).fetchone()
    if existing is None and row.get("tmdb_id"):
        existing = con.execute(
            "SELECT item_id FROM titles WHERE tmdb_id = ?", [row["tmdb_id"]]
        ).fetchone()
    if existing:
        return int(existing[0])

    next_id = con.execute("SELECT coalesce(max(item_id), -1) + 1 FROM titles").fetchone()[0]
    payload = {
        "item_id": int(next_id),
        "imdb_id": row.get("imdb_id"),
        "tmdb_id": row.get("tmdb_id"),
        "movielens_id": None,
        "kind": row.get("kind") or "movie",
        "title": row.get("title"),
        "original_title": row.get("original_title"),
        "year": row.get("year"),
        "end_year": None,
        "runtime": row.get("runtime"),
        "genres": row.get("genres") or [],
        "language": row.get("language") or "xx",
        "languages": row.get("languages") or [],
        "countries": row.get("countries") or [],
        "imdb_rating": None,
        "imdb_votes": None,
        "tmdb_rating": row.get("tmdb_rating"),
        "tmdb_votes": row.get("tmdb_votes"),
        "popularity": row.get("popularity"),
        "directors": row.get("directors") or [],
        "writers": row.get("writers") or [],
        "cast_names": row.get("cast_names") or [],
        "keywords": row.get("keywords") or [],
        "overview": row.get("overview"),
        "tagline": row.get("tagline"),
        "poster_path": row.get("poster_path"),
        "adult": bool(row.get("adult")),
        # TMDB's vote pool is smaller and differently calibrated from IMDb's,
        # so it gets a correspondingly weaker prior weight.
        "quality": quality_prior(row.get("tmdb_rating"), row.get("tmdb_votes"), prior_weight=400.0),
    }
    con.execute(
        f"INSERT INTO titles ({', '.join(payload)}, enriched_at) "
        f"VALUES ({', '.join('?' for _ in payload)}, now())",
        list(payload.values()),
    )
    return int(next_id)


def stamp_existing_keywords(con) -> int:
    """Treat keywords already stored as fetched.

    A catalogue enriched before ``keywords_at`` existed has keywords but no
    marker, and without this every one of those titles is requested again —
    tens of thousands of needless API calls on the first run after upgrading.

    Returns how many rows were marked.
    """
    before = con.execute(
        "SELECT count(*) FROM titles WHERE keywords_at IS NULL "
        "AND keywords IS NOT NULL AND len(keywords) > 0"
    ).fetchone()[0]
    con.execute(
        "UPDATE titles SET keywords_at = now() "
        "WHERE keywords_at IS NULL AND keywords IS NOT NULL AND len(keywords) > 0"
    )
    return int(before)


def keyword_targets(con, top: int) -> list[tuple[int, int, str]]:
    """Titles among the ``top`` most-voted that still have no keywords.

    ``top`` is a coverage target, not a batch size: the claim is "the N
    most-voted titles should have keywords". Ranking first and filtering
    second is what makes that true — filtering first would rank whatever is
    left, so a resumed run would march on to the next N instead of finishing
    the same N.
    """
    rows = con.execute(
        """
        SELECT item_id, tmdb_id, kind FROM (
            SELECT item_id, tmdb_id, kind, keywords_at,
                   row_number() OVER (ORDER BY imdb_votes DESC NULLS LAST) AS rank
            FROM titles WHERE tmdb_id IS NOT NULL
        ) WHERE rank <= ? AND keywords_at IS NULL
        """,
        [top],
    ).fetchall()
    return [(int(item_id), int(tmdb_id), kind) for item_id, tmdb_id, kind in rows]
