"""Measuring whether any of this works.

`stats` reports what exists, `audit` measures the engine against the user's
own history, and `eval` benchmarks it against baselines on held-out
MovieLens users."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.table import Table

from .. import pipeline, store
from ..config import has_tmdb
from ..engine import Engine
from ..errors import IntegrityRefusal
from ..manifests import write as write_manifest
from ..render import console, tables
from ..render import fail as _fail
from ..render import toned as _toned
from ._apps import app
from ._shared import require_catalog


@app.command()
def stats() -> None:
    """Show what exists, how much the engine knows, and what to run next."""
    engine = Engine()
    present = engine.artifacts_present
    table = Table("component", "state")
    for name, ok in present.items():
        table.add_row(name.replace("_", " "), "[green]ready[/green]" if ok else "[red]missing[/red]")
    table.add_row("tmdb credentials", "[green]set[/green]" if has_tmdb() else "[yellow]absent[/yellow]")
    console.print(table)

    if not pipeline.catalogue_exists():
        console.print("\n[bold]next:[/bold] [cyan]ent setup[/cyan]")
        return
    with store.session(read_only=True) as con:
        c = store.counts(con)
        langs = con.execute(
            # Tiebreak on language, so equal counts do not reorder between runs.
            "SELECT language, count(*) n FROM titles GROUP BY 1 "
            "ORDER BY n DESC, language ASC LIMIT 12"
        ).fetchall()
        cover = engine.coverage(con)

    if present["fused_space"]:
        share = pipeline.coverage_is_stale(cover)
        if share is not None:
            console.print(
                f"\n[yellow]item space covers {cover['fused']:,} of "
                f"{cover['catalog']:,} titles ({share:.1%}).[/yellow] "
                "The rest cannot be recommended. Run [cyan]ent data embed[/cyan] "
                "then [cyan]ent data fuse[/cyan]."
            )

    t2 = Table("metric", "value")
    for key, value in c.items():
        t2.add_row(key, f"{value:,}")
    console.print(t2)
    console.print(tables.language_histogram(langs))
    step = pipeline.next_step(present, c["ratings"])
    console.print(f"\n[bold]next:[/bold] [cyan]{step.command}[/cyan]" + (f"    [dim]({step.note})[/dim]" if step.note else ""))


@app.command()
def audit(
    interval: float = typer.Option(0.90, help="Nominal coverage of the predictive interval."),
) -> None:
    """Measure whether the engine is actually learning *you*, on your own history.

    Walks your verdicts in order, refits on everything before each one and
    predicts it blind. Every number here comes from a model that had not seen
    the answer.
    """
    from ..evaluation.prequential import MIN_VERDICTS
    from ..evaluation.prequential import readings as prequential_readings
    from ..evaluation.prequential import run as prequential

    require_catalog()
    engine = Engine()
    with store.session(read_only=True) as con:
        rows = con.execute(
            """
            SELECT item_id, value, ts FROM events
            WHERE kind = 'rate' AND value IS NOT NULL ORDER BY ts
            """
        ).fetchall()
        fs = engine.features(con)

    seen: set[int] = set()
    items, rewards = [], []
    for item_id, value, _ in rows:
        iid = int(item_id)
        if iid in seen or iid not in fs.index:
            continue
        seen.add(iid)
        items.append(iid)
        rewards.append(float(value) / 10.0)

    if len(items) < MIN_VERDICTS:
        _fail(
            f"only {len(items)} verdicts — this needs at least {MIN_VERDICTS} "
            "to say anything honest"
        )

    res = prequential(fs, items, rewards, interval=interval)

    table = Table("measure", "value", "reading")
    for row in prequential_readings(res, interval=interval, n_verdicts=len(items)):
        table.add_row(row.measure, row.value, _toned(row.reading, row.tone))
    console.print(table)
    console.print(
        "[dim]every prediction above was made by a model that had not seen that verdict[/dim]"
    )

    _off_policy_report(engine, fs)


def _off_policy_report(engine: Engine, fs) -> None:
    """Render the off-policy check.

    The measurement lives in evaluation.offpolicy; this decides how to phrase
    it. Hedged deliberately — it is the weaker of the two numbers `ent audit`
    prints, and reads as authoritative if presented plainly.
    """
    from ..evaluation import offpolicy
    from ..evaluation.prequential import MIN_LOGGED

    with store.session(read_only=True) as con:
        result = offpolicy.estimate(con, engine, fs)

    if result.status in ("not-enough-data", "no-model"):
        console.print(
            f"\n[dim]off-policy check: {result.n_usable} logged recommendations with an "
            f"outcome; needs {MIN_LOGGED} before the estimate means anything[/dim]"
        )
        return
    if result.status == "item-space-changed":
        console.print("\n[dim]off-policy check skipped: the item space has changed[/dim]")
        return
    if result.status != "ok":
        return

    verdict = "[green]better[/green]" if result.better else "[yellow]no better[/yellow]"
    console.print(
        f"\n[bold]off-policy estimate[/bold] [dim](weaker evidence; indicative only)[/dim]\n"
        f"  slates actually shown scored  {result.logged_value * 10:.2f}/10\n"
        f"  today's model would have      {result.estimate * 10:.2f}/10   {verdict}\n"
        f"  [dim]over {result.n_usable} logged recommendations you later rated[/dim]"
    )


@app.command("eval")
def evaluate(
    users: int = typer.Option(300, help="How many held-out MovieLens users to replay."),
    budget: int = typer.Option(30, help="Answered questions each simulated user gives."),
    elicitation: str = typer.Option(
        "v-optimal",
        help="Question-selection criterion: v-optimal | d-optimal | random (control).",
    ),
    out: Path | None = typer.Option(None, help="Write the full report as JSON."),
) -> None:
    """Benchmark the engine against baselines on held-out MovieLens users."""
    from ..evaluation import integrity
    from ..evaluation.simulate import ARMS, SimConfig, load_user_histories, paired_bootstrap, run

    require_catalog()
    engine = Engine()
    with store.session(read_only=True) as con:
        fs = engine.features(con)
        meta = engine.meta(con)
        ml_map = dict(
            con.execute(
                "SELECT movielens_id, item_id FROM titles WHERE movielens_id IS NOT NULL"
            ).fetchall()
        )

    item_of_ml = {int(k): int(v) for k, v in ml_map.items() if int(v) in fs.index}
    try:
        held = integrity.benchmark_users()
    except IntegrityRefusal as exc:
        _fail(str(exc))
    cfg = SimConfig(n_users=users, budget=budget)
    histories = load_user_histories(item_of_ml, users, cfg.seed, eligible=held)
    if not histories:
        _fail("no usable MovieLens histories among the held-out users — "
              "is the catalogue too narrow?")
    console.print(
        f"[dim]{len(histories)} simulated users, drawn from {len(held):,} held out, "
        f"budget {budget} answers[/dim]"
    )

    results = run(fs, meta, histories, cfg, elicitation=elicitation)
    evaluation_manifest = write_manifest(
        "offline-evaluation",
        {
            "config": cfg.__dict__,
            "n_users": len(histories),
            "elicitation": elicitation,
            "arms": list(results),
        },
    )

    table = Table("arm", "NDCG@10", "P@10", "MAP@10", "MRR@10", "novelty", "diversity", "serend.")
    for name, res in results.items():
        s = res.summary()
        if not s:
            continue
        style = "bold green" if name == "entertainer" else ""
        table.add_row(
            f"[{style}]{name}[/{style}]" if style else name,
            f"{s['ndcg@10']:.4f} ±{s['ndcg@10_se']:.4f}",
            f"{s['precision@10']:.4f}",
            f"{s['map@10']:.4f}",
            f"{s['mrr@10']:.4f}",
            f"{s['novelty']:.2f}",
            f"{s['diversity']:.3f}",
            f"{s['serendipity']:.3f}",
        )
    console.print(table)

    if "entertainer" in results:
        console.print("\n[bold]paired bootstrap vs each baseline (NDCG@10)[/bold]")
        for name in ARMS:
            if name == "entertainer" or name not in results:
                continue
            diff, p = paired_bootstrap(results["entertainer"], results[name])
            verdict = "[green]significant[/green]" if p < 0.05 else "[yellow]not significant[/yellow]"
            console.print(f"  vs {name:<18} Δ={diff:+.4f}  p={p:.4f}  {verdict}")

    if out:
        payload = {
            "config": cfg.__dict__,
            "n_users": len(histories),
            "elicitation": elicitation,
            "arms": {name: res.summary() for name, res in results.items()},
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[green]report -> {out}[/green]")
    console.print(f"[dim]evaluation manifest: {evaluation_manifest['path']}[/dim]")
