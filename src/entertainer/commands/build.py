"""Building and maintaining the catalogue.

The full build has an eight-stage, roughly three-hour critical path, so
each stage is also its own command: a run that dies at stage six is resumed
rather than restarted."""

from __future__ import annotations

import numpy as np
import typer
from rich.panel import Panel

from .. import pipeline, store
from ..build_events import Reporter
from ..config import has_tmdb
from ..engine import Engine
from ..manifests import write as write_manifest
from ..pipeline import preflight as pipeline_preflight
from ..render import console, tables
from ..render import fail as _fail
from ._apps import app, data_app
from ._shared import require_catalog


@app.command()
def setup(
    skip_enrich: bool = typer.Option(False, help="Skip the TMDB enrichment pass."),
    enrich_limit: int = typer.Option(0, help="Enrich only the N most-voted titles (0 = all)."),
    min_votes: int = typer.Option(50, help="Flat IMDb vote floor at build time."),
    floor_scale: float = typer.Option(1.0, help="<1 widens the catalogue, >1 narrows it."),
) -> None:
    """Run everything: download, build, enrich, prune, embed, factorise, fuse, prior."""
    from ..data import catalog, download

    settings = pipeline.BuildSettings(
        skip_enrich=skip_enrich,
        enrich_limit=enrich_limit,
        min_votes=min_votes,
        floor_scale=floor_scale,
    )
    info = pipeline_preflight(require_tmdb=not settings.skip_enrich)
    reporter = Reporter(settings={
        "skip_enrich": settings.skip_enrich,
        "enrich_limit": settings.enrich_limit,
        "min_votes": settings.min_votes,
        "floor_scale": settings.floor_scale,
        "free_bytes": info["free_bytes"],
    })
    console.print(f"[dim]preflight passed: {info['free_bytes'] / 1024**3:.1f} GiB free[/dim]")
    console.rule("[bold]1/8 downloading source data")
    with reporter.stage("sources"):
        download.fetch_imdb()
        download.fetch_movielens()

    console.rule("[bold]2/8 building catalogue")
    with reporter.stage("catalogue"):
        n = catalog.build_base(min_votes=settings.min_votes)
    console.print(f"[green]{n:,} titles[/green]")

    if not settings.skip_enrich and has_tmdb():
        console.rule("[bold]3/8 enriching from TMDB")
        with reporter.stage("tmdb"):
            data_enrich(
                limit=settings.enrich_limit or None,
                concurrency=settings.concurrency,
                keywords=False,
            )
            data_keywords(top=settings.keyword_target, concurrency=settings.concurrency)
    else:
        console.rule("[bold]3/8 TMDB enrichment skipped")
        reporter.skip("tmdb", "disabled by --skip-enrich or no TMDB credentials")

    console.rule("[bold]4/8 pruning by language")
    with reporter.stage("prune"):
        data_prune(scale=settings.floor_scale, dry_run=False)
    console.rule("[bold]5/8 encoding item text")
    with reporter.stage("embeddings"):
        data_embed(batch_size=settings.encode_batch, limit=None, fresh=False)
    console.rule("[bold]6/8 factorising MovieLens")
    with reporter.stage("cf"):
        data_cf(
            factors=settings.cf_factors,
            iterations=settings.cf_iterations,
            holdout=settings.cf_holdout,
            signal=settings.cf_signal,
        )
    console.rule("[bold]7/8 fusing item space")
    with reporter.stage("fusion"):
        data_fuse(dim=settings.fusion_dim)
    console.rule("[bold]8/8 learning the population prior")
    with reporter.stage("prior"):
        data_prior(
            max_users=settings.prior_max_users, shrinkage=settings.prior_shrinkage
        )

    with store.session(read_only=True) as con:
        counts = store.counts(con)
    manifest = write_manifest(
        "build",
        {
            "counts": counts,
            "min_votes": settings.min_votes,
            "floor_scale": settings.floor_scale,
        },
    )
    reporter.complete(manifest["id"])
    console.print(f"[dim]build manifest: {manifest['path']}[/dim]")

    console.print(Panel.fit("[bold green]ready[/bold green]\nnext: [cyan]ent onboard[/cyan]"))


@app.command("preflight")
def preflight_command() -> None:
    """Check local storage and TMDB configuration before a full build."""
    try:
        info = pipeline_preflight()
    except RuntimeError as exc:
        _fail(str(exc))
    console.print(
        f"[green]ready[/green] — {info['free_bytes'] / 1024**3:.1f} GiB free at {info['data_dir']}"
    )


@data_app.command("fetch")
def data_fetch() -> None:
    """Download the IMDb and MovieLens bulk datasets."""
    from ..data import download

    download.fetch_imdb()
    download.fetch_movielens()
    console.print("[green]downloaded[/green]")


@data_app.command("prune")
def data_prune(
    scale: float = typer.Option(1.0, help="<1 keeps more obscure titles, >1 fewer."),
    dry_run: bool = typer.Option(False, help="Report what would go, change nothing."),
) -> None:
    """Apply per-language vote floors, after TMDB has supplied the languages."""
    from ..data import catalog

    if not dry_run:
        catalog.recalibrate_quality()
    before, after = catalog.prune_by_language(scale=scale, dry_run=dry_run)
    verb = "would remove" if dry_run else "removed"
    console.print(f"[green]{before:,} -> {after:,}[/green] ({verb} {before - after:,})")
    if not dry_run:
        console.print(tables.language_histogram(catalog.language_histogram(25)))


@data_app.command("build")
def data_build(min_votes: int = typer.Option(50, help="Flat IMDb vote floor at build time.")) -> None:
    """Build the catalogue table from the bulk datasets."""
    from ..data import catalog

    n = catalog.build_base(min_votes=min_votes)
    console.print(f"[green]{n:,} titles[/green]")
    console.print(tables.language_histogram(catalog.language_histogram(20)))


@data_app.command("enrich")
def data_enrich(
    limit: int | None = typer.Option(None, help="Only the N most-voted unenriched titles."),
    concurrency: int = typer.Option(40),
    keywords: bool = typer.Option(True, help="Fetch the keyword vocabulary (doubles requests)."),
) -> None:
    """Fill in synopses, keywords and languages from TMDB."""
    from ..data import catalog, tmdb

    if not has_tmdb():
        _fail("no TMDB credentials — put TMDB_BEARER or TMDB_API_KEY in .env")
    pending = catalog.pending_enrichment(limit)
    if not pending:
        console.print("[green]nothing left to enrich[/green]")
        return
    console.print(f"[dim]{len(pending):,} titles to enrich[/dim]")

    con = store.connect()
    written = {"n": 0}

    def flush(batch):
        catalog.apply_enrichment(con, batch)
        written["n"] += len(batch)

    try:
        tmdb.enrich(pending, concurrency=concurrency, keywords=keywords, on_batch=flush)
    finally:
        con.close()
    console.print(f"[green]enriched {written['n']:,} titles[/green]")


@data_app.command("keywords")
def data_keywords(
    top: int = typer.Option(
        150_000, help="Ensure the N most-voted titles have keywords. Idempotent."
    ),
    concurrency: int = typer.Option(40),
) -> None:
    """Backfill TMDB keywords for the titles most likely to be encountered."""
    from ..data import catalog, tmdb

    if not has_tmdb():
        _fail("no TMDB credentials")
    con = store.connect()
    catalog.stamp_existing_keywords(con)
    targets = catalog.keyword_targets(con, top)
    if not targets:
        console.print("[green]nothing to backfill[/green]")
        con.close()
        return
    console.print(f"[dim]{len(targets):,} titles need keywords[/dim]")

    written = {"n": 0}

    def flush(batch):
        # keywords_at is stamped even when the list comes back empty: "TMDB
        # has none for this title" is a result, not a failure to fetch.
        con.executemany(
            "UPDATE titles SET keywords = ?, keywords_at = now() WHERE item_id = ?",
            [(kws, item_id) for item_id, kws in batch],
        )
        written["n"] += len(batch)

    try:
        tmdb.backfill_keywords(targets, concurrency=concurrency, on_batch=flush)
    finally:
        con.close()
    console.print(f"[green]keywords for {written['n']:,} titles[/green]")


@data_app.command("embed")
def data_embed(
    batch_size: int = typer.Option(64),
    limit: int | None = typer.Option(None),
    fresh: bool = typer.Option(False, help="Discard cached shards and re-encode everything."),
) -> None:
    """Encode every item card with the multilingual text encoder.

    Resumable: completed shards are written as they finish, so an interrupted
    run picks up where it stopped rather than repeating forty minutes of GPU
    work.
    """
    from ..models import encoder
    from ..models.itemcard import build_card

    require_catalog()
    con = store.connect(read_only=True)
    cur = con.execute(
        "SELECT item_id, title, original_title, year, kind, language, runtime, genres, "
        "keywords, directors, cast_names, overview, tagline FROM titles "
        "ORDER BY item_id" + (f" LIMIT {int(limit)}" if limit else "")
    )
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]
    con.close()
    if not rows:
        _fail("catalogue is empty")

    if fresh:
        console.print(f"[dim]cleared {encoder.clear_shards()} cached shards[/dim]")

    ids = np.array([r["item_id"] for r in rows], dtype=np.int32)
    cards = [build_card(r) for r in rows]
    console.print(f"[dim]{len(cards):,} item cards[/dim]")
    console.print(Panel.fit(cards[0], title="example item card", border_style="dim"))

    ids, mat = encoder.encode_resumable(ids, cards, batch_size=batch_size)
    encoder.save(ids, mat)
    console.print(f"[green]embeddings: {mat.shape}[/green]")


@data_app.command("cf")
def data_cf(
    factors: int = typer.Option(192),
    iterations: int = typer.Option(20),
    holdout: int = typer.Option(
        2000, help="MovieLens users withheld from training, reserved for offline evaluation."
    ),
    signal: str = typer.Option(
        "watched", help="watched (all ratings, graded confidence) | liked (>=3.5 only)."
    ),
) -> None:
    """Factorise the MovieLens co-consumption matrix."""

    from ..evaluation import integrity
    from ..models import cf

    ratings = cf.load_ratings()
    users = np.sort(ratings["userId"].unique().to_numpy())
    held = integrity.choose_holdout(users, holdout)

    ids, item_factors = cf.fit(
        factors=factors, iterations=iterations, holdout_users=held, signal=signal
    )
    cf.save(ids, item_factors)
    if held is not None:
        integrity.save_holdout(held)
    console.print(f"[green]CF factors: {item_factors.shape}[/green]")


@data_app.command("prior")
def data_prior(
    max_users: int = typer.Option(20_000, help="MovieLens users to fit taste vectors for."),
    shrinkage: float = typer.Option(0.15, help="Pull the covariance towards a scaled identity."),
) -> None:
    """Learn what human taste vectors look like, to use as the cold-start prior."""

    from ..evaluation import integrity
    from ..models import fusion
    from ..models.population import fit as fit_prior

    if not fusion.exists():
        _fail("no fused item space — run `ent data fuse` first")

    engine = Engine()
    with store.session(read_only=True) as con:
        fs = engine.features(con)
        ml = dict(
            con.execute(
                "SELECT movielens_id, item_id FROM titles WHERE movielens_id IS NOT NULL"
            ).fetchall()
        )
    item_of_ml = {int(k): int(v) for k, v in ml.items() if int(v) in fs.index}
    if not item_of_ml:
        _fail("no MovieLens identities in the catalogue — rebuild after downloading ml-32m")

    held = integrity.load_holdout()

    prior = fit_prior(fs, item_of_ml, holdout_users=held, max_users=max_users,
                      shrinkage=shrinkage)
    prior.save()
    console.print(
        f"[green]population prior from {prior.n_users:,} users[/green] "
        f"({prior.dim} dimensions)"
    )


@data_app.command("fuse")
def data_fuse(dim: int = typer.Option(192)) -> None:
    """Fuse the content and collaborative towers into one latent space."""
    from ..models import cf, encoder, fusion

    if not encoder.exists():
        _fail("no content embeddings — run `ent data embed`")
    if not cf.exists():
        _fail("no CF factors — run `ent data cf`")

    ids, content = encoder.load()
    cf_ids, cf_factors = cf.load()

    con = store.connect(read_only=True)
    ml = dict(
        con.execute(
            "SELECT item_id, movielens_id FROM titles WHERE movielens_id IS NOT NULL"
        ).fetchall()
    )
    con.close()

    art = fusion.build(ids, content, cf_ids, cf_factors, {int(k): int(v) for k, v in ml.items()}, dim=dim)
    fusion.save(art)
    console.print(
        f"[green]fused space: {art.space.shape}[/green] "
        f"(CF coverage {art.cf_coverage:.1%}, imputation R²={art.cf_r2:.3f})"
    )


# --- taste input ------------------------------------------------------------
