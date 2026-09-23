"""Leakage-resistant personal validation for the local rating app."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.linear_model import RidgeCV

from .. import store
from ..engine import NEGATIVE_REWARD, NEGATIVE_WEIGHT, Engine
from ..manifests import write as write_manifest
from ..models.taste import verdict_to_reward

LIKE_AT = 0.75
PRELIMINARY_CASES = 30
DECISION_CASES = 100
MIN_RANKING_CANDIDATES = 20


@dataclass(frozen=True)
class Prediction:
    score: float
    std: float
    like_probability: float


def _probability(mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std = np.maximum(std, 1e-6)
    return stats.norm.sf((LIKE_AT - mean) / std)


def _ridge_predictions(engine: Engine, con, item_ids: list[int]) -> np.ndarray:
    fs = engine.features(con)
    ids, rewards, _, _ = engine.labels(con)
    if len(ids) < 3:
        raise ValueError("at least three explicit verdicts are needed before sealing validation")
    known = set(ids.tolist())
    rng = np.random.default_rng(0)
    rows = rng.choice(len(fs.item_ids), size=min(1000, len(fs.item_ids)), replace=False)
    rows = np.array([r for r in rows if int(fs.item_ids[r]) not in known], dtype=np.int64)
    X = np.vstack([fs.vectors_for(ids), fs.matrix[rows]])
    y = np.concatenate([rewards, np.full(rows.size, NEGATIVE_REWARD)])
    weights = np.concatenate([np.ones(len(rewards)), np.full(rows.size, NEGATIVE_WEIGHT)])
    ridge = RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0)).fit(X, y, sample_weight=weights)
    return ridge.predict(fs.vectors_for(item_ids))


def predict(engine: Engine, item_id: int) -> dict[str, float | bool]:
    """A display-safe prediction for one catalogue title.

    The stored score remains on the model's 0..1 scale; clamping happens only
    at the API edge so ranking arithmetic remains untouched elsewhere.
    """
    with store.session(read_only=True) as con:
        fs = engine.features(con)
        if item_id not in fs.index:
            raise ValueError("this title is not in the current item space")
        model = engine.fit(con, save=False)
        if model is None:
            raise ValueError("at least three explicit verdicts are needed before prediction")
        mean, std = model.predict(fs.vectors_for([item_id]))
        ridge = _ridge_predictions(engine, con, [item_id])
    probability = float(_probability(mean, std)[0])
    score = float(mean[0])
    deviation = float(std[0])
    return {
        "score": min(10.0, max(0.0, score * 10.0)),
        "interval_low": min(10.0, max(0.0, (score - 1.645 * deviation) * 10.0)),
        "interval_high": min(10.0, max(0.0, (score + 1.645 * deviation) * 10.0)),
        "like_probability": probability,
        "likely_like": probability >= 0.5,
        "ridge_score": min(10.0, max(0.0, float(ridge[0]) * 10.0)),
    }


def seal(engine: Engine, item_ids: list[int]) -> dict:
    """Persist predictions before any validation verdict can be recorded."""
    unique = list(dict.fromkeys(int(i) for i in item_ids))
    if len(unique) < MIN_RANKING_CANDIDATES:
        raise ValueError(
            f"seal at least {MIN_RANKING_CANDIDATES} watched-but-unrated titles; "
            "a Top-10 comparison needs more candidates than it displays"
        )
    with store.session() as con:
        rated = {int(r[0]) for r in con.execute("SELECT DISTINCT item_id FROM events WHERE kind = 'rate'").fetchall()}
        existing = {int(r[0]) for r in con.execute("SELECT item_id FROM validation_cases").fetchall()}
        invalid = [i for i in unique if i in rated or i in existing]
        missing = [i for i in unique if not con.execute("SELECT 1 FROM titles WHERE item_id = ?", [i]).fetchone()]
        if invalid:
            raise ValueError("validation titles must not already have an explicit rating or prior case")
        if missing:
            raise ValueError("every validation title must be in the local catalogue")
        # The evidence snapshot stores predictions itself; persisting a
        # transient taste.npz here is unnecessary and can point at a stale
        # data root after a test/profile switch.
        model = engine.fit(con, save=False)
        if model is None:
            raise ValueError("at least three explicit verdicts are needed before sealing validation")
        fs = engine.features(con)
        mean, std = model.predict(fs.vectors_for(unique))
        ridge = _ridge_predictions(engine, con, unique)
        full_p = _probability(mean, std)
        # Ridge has no posterior variance; use its residual-scale-free score as
        # a calibrated ranking probability only, not a confidence interval.
        ridge_p = 1.0 / (1.0 + np.exp(-(ridge - LIKE_AT) * 5.0))
        manifest = write_manifest("personal-validation", {"item_ids": unique, "n_labels": int(len(engine.labels(con)[0]))})
        batch_id = f"validation-{uuid.uuid4().hex}"
        full_rank = [unique[i] for i in np.argsort(-mean)]
        ridge_rank = [unique[i] for i in np.argsort(-ridge)]
        con.execute(
            "INSERT INTO validation_batches VALUES (?, now(), ?, ?, ?, ?)",
            [batch_id, json.dumps(unique), json.dumps(full_rank), json.dumps(ridge_rank), manifest["id"]],
        )
        cases = []
        for i, item_id in enumerate(unique):
            case_id = f"case-{uuid.uuid4().hex}"
            con.execute(
                "INSERT INTO validation_cases (case_id, batch_id, item_id, sealed_at, full_score, full_std, full_like_prob, ridge_score, ridge_like_prob) VALUES (?, ?, ?, now(), ?, ?, ?, ?, ?)",
                [case_id, batch_id, item_id, float(mean[i]), float(std[i]), float(full_p[i]), float(ridge[i]), float(ridge_p[i])],
            )
            cases.append({"case_id": case_id, "item_id": item_id})
    return {"batch_id": batch_id, "cases": cases, "manifest_id": manifest["id"]}


def reveal(engine: Engine, case_id: str, verdict: str) -> dict:
    with store.session() as con:
        row = con.execute("SELECT item_id, status FROM validation_cases WHERE case_id = ?", [case_id]).fetchone()
        if not row:
            raise ValueError("unknown validation case")
        if row[1] != "sealed":
            raise ValueError("a sealed prediction can only be revealed once")
        if verdict == "unseen":
            con.execute("UPDATE validation_cases SET status = 'unseen', actual_verdict = ?, revealed_at = now() WHERE case_id = ?", [verdict, case_id])
            return {"ok": True, "status": "unseen"}
        reward = verdict_to_reward(verdict)
        engine.record(con, int(row[0]), verdict, source="validation")
        con.execute(
            "UPDATE validation_cases SET status = 'revealed', actual_reward = ?, actual_verdict = ?, revealed_at = now() WHERE case_id = ?",
            [reward, verdict, case_id],
        )
    return {"ok": True, "status": "revealed"}


def open_cases() -> list[dict]:
    """Sealed cases still awaiting a verdict, with the titles they refer to.

    Without this the sealed pool is unrecoverable. The browser held the
    item_id-to-case_id mapping in memory only, and a sealed title is refused
    by the ordinary rating endpoint by design — so a page reload stranded
    every case permanently, with no way to reveal it and no way to rate it.
    """
    from .. import store

    with store.session(read_only=True) as con:
        rows = con.execute(
            """
            SELECT c.case_id, c.item_id, c.batch_id, c.sealed_at, t.title, t.year, t.kind
            FROM validation_cases c JOIN titles t USING (item_id)
            WHERE c.status = 'sealed'
            ORDER BY c.sealed_at, c.case_id
            """
        ).fetchall()
    return [
        {
            "case_id": case_id,
            "item_id": int(item_id),
            "batch_id": batch_id,
            "sealed_at": str(sealed_at),
            "title": title,
            "year": year,
            "kind": kind,
        }
        for case_id, item_id, batch_id, sealed_at, title, year, kind in rows
    ]


def _interval(values: np.ndarray, seed: int = 0) -> list[float] | None:
    if values.size < 2:
        return None
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, size=(5000, values.size))].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def summary() -> dict:
    """Report completed hidden pools only; partial pools cannot flatter a ranking."""
    with store.session(read_only=True) as con:
        batches = con.execute("SELECT batch_id, full_ranking, ridge_ranking FROM validation_batches").fetchall()
        cases = con.execute("SELECT batch_id, item_id, full_score, full_std, full_like_prob, ridge_score, ridge_like_prob, actual_reward FROM validation_cases WHERE status = 'revealed'").fetchall()
    by_batch: dict[str, dict[int, tuple]] = {}
    for batch_id, item_id, *values in cases:
        by_batch.setdefault(batch_id, {})[int(item_id)] = tuple(float(v) for v in values)
    full_top10, ridge_top10, paired_lift = [], [], []
    full_top10_labels, ridge_top10_labels = [], []
    full_mae, ridge_mae, full_brier = [], [], []
    completed = 0
    for batch_id, full_json, ridge_json in batches:
        full_rank, ridge_rank = json.loads(full_json), json.loads(ridge_json)
        rows = by_batch.get(batch_id, {})
        if set(full_rank) != set(rows):
            continue
        completed += len(rows)
        actual = {item: rows[item][-1] for item in rows}
        k = min(10, len(full_rank))
        # Ranking metrics are one observation per sealed candidate slate. This
        # keeps the model arms paired on exactly the same titles and prevents a
        # 2-title pool from masquerading as a Top-10 result.
        full_rate = float(np.mean([actual[i] >= LIKE_AT for i in full_rank[:k]]))
        ridge_rate = float(np.mean([actual[i] >= LIKE_AT for i in ridge_rank[:k]]))
        full_top10.append(full_rate)
        ridge_top10.append(ridge_rate)
        paired_lift.append(full_rate - ridge_rate)
        full_top10_labels.extend(float(actual[i] >= LIKE_AT) for i in full_rank[:k])
        ridge_top10_labels.extend(float(actual[i] >= LIKE_AT) for i in ridge_rank[:k])
        for _item, (score, _std, probability, ridge_score, _ridge_p, reward) in rows.items():
            full_mae.append(abs(score - reward))
            ridge_mae.append(abs(ridge_score - reward))
            full_brier.append((probability - float(reward >= LIKE_AT)) ** 2)
    hit_delta = np.asarray(paired_lift)
    decision = "collecting"
    if completed >= DECISION_CASES:
        ci = _interval(hit_delta)
        decision = "keep-full" if ci and ci[0] > 0 else "simplify-to-ridge"
    return {
        "completed_cases": completed,
        "status": "preliminary" if PRELIMINARY_CASES <= completed < DECISION_CASES else decision,
        "full": {"top10_hit_rate": float(np.mean(full_top10)) if full_top10 else None, "top10_hit_rate_ci95": _interval(np.asarray(full_top10_labels)), "mae": float(np.mean(full_mae)) if full_mae else None, "mae_ci95": _interval(np.asarray(full_mae)), "brier": float(np.mean(full_brier)) if full_brier else None},
        "ridge": {"top10_hit_rate": float(np.mean(ridge_top10)) if ridge_top10 else None, "top10_hit_rate_ci95": _interval(np.asarray(ridge_top10_labels)), "mae": float(np.mean(ridge_mae)) if ridge_mae else None, "mae_ci95": _interval(np.asarray(ridge_mae))},
        "top10_lift_ci95": _interval(hit_delta),
        "decision": decision,
    }
