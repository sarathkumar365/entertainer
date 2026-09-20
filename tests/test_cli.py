"""End-to-end CLI tests against a synthetic data directory.

The CLI is the largest untested surface in the project and the only part the
user actually touches. These build a miniature catalogue with real embeddings
and a real fused space in a temp directory, then drive the commands the way a
person would: name a film, say what you thought, ask for something to watch.

No network, no GPU, no downloads.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest
from typer.testing import CliRunner

TITLES = [
    # (imdb_id, title, original, year, kind, language, votes, rating)
    ("tt10000001", "Kumbalangi Nights", "Kumbalangi Nights", 2019, "movie", "ml", 40000, 8.4),
    ("tt10000002", "Jallikattu", "Jallikattu", 2019, "movie", "ml", 12000, 7.3),
    ("tt10000003", "Drishyam", "Drishyam", 2013, "movie", "ml", 35000, 8.3),
    ("tt10000004", "Drishyam", "Drishyam", 2015, "movie", "hi", 90000, 8.2),
    ("tt10000005", "Memories of Murder", "Salinui chueok", 2003, "movie", "ko", 200000, 8.1),
    ("tt10000006", "Parasite", "Gisaengchung", 2019, "movie", "ko", 900000, 8.5),
    ("tt10000007", "The Godfather", "The Godfather", 1972, "movie", "en", 2000000, 9.2),
    ("tt10000008", "Whiplash", "Whiplash", 2014, "movie", "en", 950000, 8.5),
    ("tt10000009", "Morbius", "Morbius", 2022, "movie", "en", 250000, 5.2),
    ("tt10000010", "Transformers", "Transformers", 2007, "movie", "en", 650000, 7.0),
    ("tt10000011", "Breaking Bad", "Breaking Bad", 2008, "tv", "en", 2100000, 9.5),
    ("tt10000012", "Dark", "Dark", 2017, "tv", "de", 450000, 8.7),
    ("tt10000013", "Kantara", "Kantara", 2022, "movie", "kn", 150000, 8.2),
    ("tt10000014", "96", "96", 2018, "movie", "ta", 40000, 8.5),
    ("tt10000015", "Super Deluxe", "Super Deluxe", 2019, "movie", "ta", 30000, 8.3),
    ("tt10000016", "Tumbbad", "Tumbbad", 2018, "movie", "hi", 60000, 8.2),
]

# Two planted taste groups, so a recommendation can be checked for landing in
# the right one rather than merely for not crashing.
GROUP_A = {0, 1, 2, 4, 13, 14}      # slow-burn, South Indian and Korean
GROUP_B = {8, 9}                    # loud blockbusters


@pytest.fixture()
def app_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTERTAINER_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("TMDB_API_KEY", raising=False)
    monkeypatch.delenv("TMDB_BEARER", raising=False)

    from entertainer import config, engine, store
    from entertainer.models import encoder, fusion

    for mod in (config, store, engine, encoder, fusion):
        importlib.reload(mod)
    from entertainer import cli, recommend, resolve
    from entertainer.coldstart import elicit
    from entertainer.models import features

    for mod in (features, recommend, resolve, elicit, cli):
        importlib.reload(mod)

    config.PATHS.ensure()

    con = store.connect()
    for i, (imdb_id, title, original, year, kind, lang, votes, rating) in enumerate(TITLES):
        con.execute(
            """
            INSERT INTO titles
            (item_id, imdb_id, kind, title, original_title, year, language, runtime,
             genres, keywords, directors, cast_names, imdb_rating, imdb_votes, quality,
             overview, adult)
            VALUES (?, ?, ?, ?, ?, ?, ?, 120, ?, ?, ?, ?, ?, ?, ?, ?, false)
            """,
            [
                i, imdb_id, kind, title, original, year, lang,
                ["Drama"], ["theme"], ["A Director"], ["An Actor"],
                rating, votes, rating / 10.0, f"A synopsis for {title}.",
            ],
        )
    con.close()

    # A latent space with the planted groups actually separated.
    rng = np.random.default_rng(0)
    dim = 16
    anchors = rng.normal(size=(3, dim))
    latent = np.empty((len(TITLES), dim), dtype=np.float32)
    for i in range(len(TITLES)):
        anchor = anchors[0] if i in GROUP_A else (anchors[1] if i in GROUP_B else anchors[2])
        latent[i] = anchor + 0.25 * rng.normal(size=dim)
    latent /= np.linalg.norm(latent, axis=1, keepdims=True)

    ids = np.arange(len(TITLES), dtype=np.int32)
    fusion.save(
        fusion.FusionArtifacts(
            item_ids=ids,
            space=latent,
            components=np.eye(dim, dtype=np.float32),
            block_sizes=(dim, 0),
            cf_r2=0.5,
            cf_coverage=1.0,
            pca_mean=np.zeros(dim, dtype=np.float32),
            ridge_coef=np.zeros((0, dim), dtype=np.float32),
            ridge_intercept=np.zeros(0, dtype=np.float32),
        )
    )
    encoder.save(ids, latent)
    return cli, CliRunner()


def run(cli, runner, *args, stdin: str | None = None):
    return runner.invoke(cli.app, list(args), input=stdin, catch_exceptions=False)


def teach(cli, runner, pairs):
    for title, verdict in pairs:
        res = run(cli, runner, verdict, title)
        assert res.exit_code == 0, res.output


def test_stats_reports_the_catalogue(app_env):
    cli, runner = app_env
    res = run(cli, runner, "stats")
    assert res.exit_code == 0
    assert "16" in res.output
    assert "ml" in res.output


def test_find_matches_original_titles(app_env):
    cli, runner = app_env
    res = run(cli, runner, "find", "Gisaengchung")
    assert res.exit_code == 0
    assert "Parasite" in res.output


def test_recording_a_verdict_by_name(app_env):
    cli, runner = app_env
    res = run(cli, runner, "loved", "Kumbalangi Nights")
    assert res.exit_code == 0
    assert "love" in res.output
    hist = run(cli, runner, "history")
    assert "Kumbalangi" in hist.output


def test_ambiguous_title_asks_rather_than_guessing(app_env):
    """Two films share the title 'Drishyam'. Guessing would poison the log."""
    cli, runner = app_env
    res = run(cli, runner, "loved", "Drishyam", stdin="2\n")
    assert res.exit_code == 0
    assert "which one?" in res.output
    # Both years must be offered, and exactly one recorded.
    assert "2013" in res.output and "2015" in res.output
    hist = run(cli, runner, "history")
    years = [line for line in hist.output.splitlines() if "Drishyam" in line]
    assert len(years) == 1


def test_year_removes_the_ambiguity(app_env):
    cli, runner = app_env
    res = run(cli, runner, "loved", "Drishyam (2013)")
    assert res.exit_code == 0
    assert "which one?" not in res.output


def test_recommendations_need_evidence_first(app_env):
    cli, runner = app_env
    res = run(cli, runner, "recs")
    assert res.exit_code == 1
    assert "verdict" in res.output


def test_recommendations_follow_the_planted_taste(app_env):
    cli, runner = app_env
    teach(cli, runner, [
        ("Kumbalangi Nights", "loved"),
        ("Jallikattu", "liked"),
        ("96", "loved"),
        ("Morbius", "hated"),
        ("Transformers", "hated"),
    ])
    res = run(cli, runner, "recs", "-k", "4", "--strategy", "mean")
    assert res.exit_code == 0, res.output
    # Super Deluxe and Memories of Murder sit in the liked group; the
    # blockbusters should not surface.
    assert "Morbius" not in res.output
    assert any(t in res.output for t in ("Super Deluxe", "Memories of Murder", "Drishyam"))


def test_language_filter_is_hard(app_env):
    cli, runner = app_env
    teach(cli, runner, [("Kumbalangi Nights", "loved"), ("Morbius", "hated"), ("96", "liked")])
    res = run(cli, runner, "recs", "--lang", "ko", "-k", "3", "--strategy", "mean")
    assert res.exit_code == 0, res.output
    assert "Korean" in res.output
    assert "Tamil" not in res.output


def test_series_filter(app_env):
    cli, runner = app_env
    teach(cli, runner, [("Kumbalangi Nights", "loved"), ("Morbius", "hated"), ("96", "liked")])
    res = run(cli, runner, "recs", "--series", "-k", "3", "--strategy", "mean")
    assert res.exit_code == 0, res.output
    assert "Breaking Bad" in res.output or "Dark" in res.output


def test_rated_titles_are_never_recommended_back(app_env):
    cli, runner = app_env
    teach(cli, runner, [
        ("Kumbalangi Nights", "loved"), ("Jallikattu", "loved"),
        ("96", "loved"), ("Morbius", "hated"),
    ])
    res = run(cli, runner, "recs", "-k", "10", "--strategy", "mean")
    # Only the slate's own header lines count; the explanation lines name the
    # user's own rated titles on purpose.
    headers = [ln for ln in res.output.splitlines() if ln.startswith(("◆", "◇"))]
    assert headers
    joined = " ".join(headers)
    for rated in ("Jallikattu", "Kumbalangi", "Morbius", "96 "):
        assert rated not in joined, joined


def test_predicted_scores_are_readable(app_env):
    """An unbounded linear model must not print 10.4 out of 10."""
    cli, runner = app_env
    teach(cli, runner, [
        ("Kumbalangi Nights", "loved"), ("Jallikattu", "loved"), ("96", "loved"),
        ("Drishyam (2013)", "loved"), ("Morbius", "hated"), ("Transformers", "hated"),
    ])
    res = run(cli, runner, "recs", "-k", "6", "--strategy", "mean")
    assert res.exit_code == 0, res.output
    import re

    scores = [float(m) for m in re.findall(r"predicted (\d+\.\d)/10", res.output)]
    assert scores
    assert all(0.0 <= s <= 10.0 for s in scores), scores


def test_why_explains_a_specific_title(app_env):
    cli, runner = app_env
    teach(cli, runner, [
        ("Kumbalangi Nights", "loved"), ("96", "loved"), ("Morbius", "hated"),
    ])
    res = run(cli, runner, "why", "Super Deluxe")
    assert res.exit_code == 0, res.output
    assert "predicted" in res.output


def test_taste_summarises_what_was_learned(app_env):
    cli, runner = app_env
    teach(cli, runner, [
        ("Kumbalangi Nights", "loved"), ("Jallikattu", "liked"), ("96", "loved"),
        ("Morbius", "hated"), ("Transformers", "hated"),
    ])
    res = run(cli, runner, "taste")
    assert res.exit_code == 0, res.output
    assert "verdicts" in res.output
    assert "surface preferences" in res.output


def test_similar_ignores_the_profile(app_env):
    cli, runner = app_env
    res = run(cli, runner, "similar", "Kumbalangi Nights", "-k", "3")
    assert res.exit_code == 0, res.output
    assert "closest to" in res.output


def test_forget_removes_a_verdict(app_env):
    cli, runner = app_env
    teach(cli, runner, [("Whiplash", "loved")])
    assert "Whiplash" in run(cli, runner, "history").output
    res = run(cli, runner, "forget", "Whiplash")
    assert res.exit_code == 0
    assert "Whiplash" not in run(cli, runner, "history").output


def test_bulk_import_and_unresolved_reporting(app_env, tmp_path):
    cli, runner = app_env
    path = tmp_path / "seed.txt"
    path.write_text(
        "# a comment\n"
        "Kumbalangi Nights | love\n"
        "Morbius | hate\n"
        "Whiplash\n"
        "A Film That Does Not Exist Anywhere\n",
        encoding="utf-8",
    )
    res = run(cli, runner, "bulk", str(path))
    assert res.exit_code == 0, res.output
    assert "recorded 3 verdicts" in res.output
    assert "1 unresolved" in res.output


def test_export_import_roundtrip_survives_an_item_id_change(app_env, tmp_path):
    """Item ids are positional and change on rebuild; IMDb ids do not."""
    cli, runner = app_env
    from entertainer import store

    teach(cli, runner, [("Kumbalangi Nights", "loved"), ("Morbius", "hated")])
    out = tmp_path / "profile.jsonl"
    assert run(cli, runner, "export", "--path", str(out)).exit_code == 0

    with store.session() as con:
        con.execute("DELETE FROM events")
    assert "Kumbalangi" not in run(cli, runner, "history").output

    res = run(cli, runner, "import", str(out))
    assert res.exit_code == 0, res.output
    assert "Kumbalangi" in run(cli, runner, "history").output


def test_audit_refuses_to_pronounce_on_too_little_history(app_env):
    cli, runner = app_env
    teach(cli, runner, [("Kumbalangi Nights", "loved"), ("Morbius", "hated")])
    res = run(cli, runner, "audit")
    assert res.exit_code == 1
    assert "at least 8" in res.output


def test_add_requires_credentials(app_env):
    cli, runner = app_env
    res = run(cli, runner, "add", "Some Obscure Film")
    assert res.exit_code == 1
    assert "TMDB" in res.output


def test_unknown_title_fails_cleanly(app_env):
    cli, runner = app_env
    res = run(cli, runner, "loved", "zzzz nonexistent qqqq")
    assert res.exit_code == 1
    assert "nothing matching" in res.output
