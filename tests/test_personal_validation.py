"""Guards on the blind-validation flow.

This is the measurement the project will eventually be judged on, so the
properties that make it honest need pinning. Every one of these is a way the
flow could quietly start flattering the model:

* a title that already has a verdict being admitted as a "blind" case;
* a sealed prediction moving after the outcome is known;
* an unresolved "haven't seen it" counting towards accuracy;
* a partially-revealed pool contributing a ranking metric.
"""

from __future__ import annotations

import datetime as dt
import importlib

import numpy as np
import pytest


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTERTAINER_DATA_DIR", str(tmp_path))
    from entertainer import config, engine, store

    for mod in (config, store, engine):
        importlib.reload(mod)
    from entertainer.evaluation import personal
    from entertainer.models import encoder, features, fusion

    for mod in (features, encoder, fusion, personal):
        importlib.reload(mod)
    config.PATHS.ensure()

    year = dt.date.today().year
    n = 60
    con = store.connect()
    for i in range(n):
        con.execute(
            """
            INSERT INTO titles (item_id, imdb_id, kind, title, original_title, year,
                language, runtime, genres, imdb_rating, imdb_votes, quality,
                overview, poster_path)
            VALUES (?, ?, 'movie', ?, ?, ?, ?, 120, ['Drama'], 7.5, 5000, 0.6, ?, ?)
            """,
            [i, f"tt{i:07d}", f"Film {i}", f"Film {i}", year - (i % 3),
             ["ml", "ta", "en"][i % 3], f"Synopsis {i}", f"/p{i}.jpg"],
        )
    con.close()

    rng = np.random.default_rng(0)
    latent = rng.normal(size=(n, 12)).astype(np.float32)
    latent /= np.linalg.norm(latent, axis=1, keepdims=True)
    ids = np.arange(n, dtype=np.int32)
    fusion.save(
        fusion.FusionArtifacts(
            item_ids=ids, space=latent, components=np.eye(12, dtype=np.float32),
            block_sizes=(12, 0), cf_r2=0.5, cf_coverage=1.0,
            pca_mean=np.zeros(12, dtype=np.float32),
            ridge_coef=np.eye(12, dtype=np.float32),
            ridge_intercept=np.zeros(12, dtype=np.float32),
        )
    )
    encoder.save(ids, latent)

    eng = engine.Engine()
    with store.session() as con:
        for i in range(0, 12):
            eng.record(con, i, "love" if i % 2 else "dislike", source="test")
    return eng, personal, store


def test_an_already_rated_title_cannot_become_a_blind_case(env):
    """Otherwise the model is graded on something it was trained on."""
    eng, personal, _ = env
    with pytest.raises(ValueError, match="already have an explicit rating"):
        personal.seal(eng, [0, 30, 31])


def test_a_title_cannot_be_sealed_twice(env):
    eng, personal, _ = env
    personal.seal(eng, [30, 31, 32])
    with pytest.raises(ValueError, match="prior case"):
        personal.seal(eng, [32, 33])


def test_sealing_requires_titles_that_exist(env):
    eng, personal, _ = env
    with pytest.raises(ValueError, match="in the local catalogue"):
        personal.seal(eng, [30, 99999])


def test_the_sealed_prediction_is_immutable_across_reveal(env):
    """The whole point: the prediction must predate the outcome."""
    eng, personal, store = env
    batch = personal.seal(eng, [30, 31, 32, 33])

    with store.session(read_only=True) as con:
        before = con.execute(
            "SELECT case_id, full_score, full_std, full_like_prob, ridge_score "
            "FROM validation_cases ORDER BY case_id"
        ).fetchall()

    for case in batch["cases"]:
        personal.reveal(eng, case["case_id"], "love")

    with store.session(read_only=True) as con:
        after = con.execute(
            "SELECT case_id, full_score, full_std, full_like_prob, ridge_score "
            "FROM validation_cases ORDER BY case_id"
        ).fetchall()
    assert before == after, "revealing an outcome moved the sealed prediction"


def test_revealing_records_a_real_verdict_that_trains_the_model(env):
    eng, personal, store = env
    batch = personal.seal(eng, [30, 31, 32, 33])
    with store.session(read_only=True) as con:
        before = len(store.ratings(con))

    personal.reveal(eng, batch["cases"][0]["case_id"], "love")

    with store.session(read_only=True) as con:
        after = store.ratings(con)
    assert len(after) == before + 1
    assert 30 in {int(i) for i, _ in after}


def test_a_case_can_only_be_revealed_once(env):
    eng, personal, _ = env
    batch = personal.seal(eng, [30, 31])
    case = batch["cases"][0]["case_id"]
    personal.reveal(eng, case, "love")
    with pytest.raises(ValueError, match="only be revealed once"):
        personal.reveal(eng, case, "hate")


def test_not_seen_leaves_the_case_unresolved_and_out_of_the_metrics(env):
    eng, personal, store = env
    batch = personal.seal(eng, [30, 31, 32, 33])
    for case in batch["cases"]:
        personal.reveal(eng, case["case_id"], "unseen")

    with store.session(read_only=True) as con:
        statuses = {
            r[0] for r in con.execute("SELECT status FROM validation_cases").fetchall()
        }
        rated = len(store.ratings(con))
    assert statuses == {"unseen"}
    # An unseen answer is not a verdict and must not enter training.
    assert rated == 12

    s = personal.summary()
    assert s["completed_cases"] == 0
    assert s["full"]["top10_hit_rate"] is None


def test_a_partially_revealed_pool_contributes_no_ranking_metric(env):
    """A half-revealed pool could be cherry-picked into a flattering ranking."""
    eng, personal, _ = env
    batch = personal.seal(eng, [30, 31, 32, 33, 34])
    personal.reveal(eng, batch["cases"][0]["case_id"], "love")
    personal.reveal(eng, batch["cases"][1]["case_id"], "love")

    s = personal.summary()
    assert s["completed_cases"] == 0, s
    assert s["full"]["top10_hit_rate"] is None


def test_a_completed_pool_reports_both_arms_with_intervals(env):
    eng, personal, _ = env
    batch = personal.seal(eng, [30, 31, 32, 33, 34, 35])
    for i, case in enumerate(batch["cases"]):
        personal.reveal(eng, case["case_id"], "love" if i % 2 else "dislike")

    s = personal.summary()
    assert s["completed_cases"] == 6
    for arm in ("full", "ridge"):
        assert s[arm]["top10_hit_rate"] is not None
        assert s[arm]["mae"] is not None
    assert s["top10_lift_ci95"] is not None


def test_no_verdict_is_claimed_before_the_decision_threshold(env):
    """Thirty cases cannot settle this; the status must say so."""
    eng, personal, _ = env
    batch = personal.seal(eng, [30, 31, 32, 33])
    for case in batch["cases"]:
        personal.reveal(eng, case["case_id"], "love")

    s = personal.summary()
    assert s["decision"] == "collecting"
    assert s["completed_cases"] < personal.DECISION_CASES


def test_sealing_needs_a_pool_not_a_single_title(env):
    eng, personal, _ = env
    with pytest.raises(ValueError, match="at least two"):
        personal.seal(eng, [30])
