"""Does any of this actually work?

The honest problem with building a recommender for one person is that you
cannot evaluate it on that person until months have passed. So the engine is
validated the only way it can be validated up front: by replaying strangers
through it.

MovieLens users held out of collaborative-filtering training are treated as
cold-start arrivals. The engine knows nothing about them. It asks its
elicitation questions; the simulated user answers from their real rating
history, or says "haven't seen it" when the engine asks about something they
never rated. After a fixed answer budget the engine ranks everything it has
not been told about, and that ranking is scored against the user's held-out
ratings.

Two properties make this a fair test rather than a flattering one.

*No leakage.* Held-out users are excluded from the ALS fit, so the item
factors the engine scores with have never seen their opinions. Their
elicitation answers and their evaluation ratings are disjoint splits of their
own history.

*Same information to every arm.* Each baseline receives exactly the same
answered questions that the full model received, so a win cannot come from
the model having simply asked more or better-targeted questions than a
baseline was allowed to. The elicitation strategy is measured separately, by
varying the budget.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from ..coldstart import elicit
from ..models import cf as cf_mod
from ..models.features import FeatureSpace
from ..models.taste import fit as fit_taste
from . import metrics

console = Console()

# A rating at or above this on MovieLens' 0.5..5 scale counts as a hit.
RELEVANT_AT = 4.0
MIN_HISTORY = 60


@dataclass
class SimConfig:
    n_users: int = 400
    budget: int = 30
    eval_fraction: float = 0.4
    top_k: int = 10
    max_asks: int = 400
    seed: int = 0
    seed_questions: int = 12
    criterion: str = "v-optimal"   # v-optimal | d-optimal


@dataclass
class ArmResult:
    name: str
    per_user: list[dict] = field(default_factory=list)

    def summary(self) -> dict[str, float]:
        return metrics.summarise(self.per_user)


def _rating_to_reward(r: float) -> float:
    """MovieLens 0.5..5 onto the engine's 0..1 reward scale."""
    return float(np.clip((r - 0.5) / 4.5, 0.0, 1.0))


def load_user_histories(
    item_of_movielens: dict[int, int],
    n_users: int,
    seed: int,
    *,
    eligible: np.ndarray,
    min_history: int = MIN_HISTORY,
) -> dict[int, dict[int, float]]:
    """Sample MovieLens users and map their ratings onto catalogue item ids.

    ``eligible`` is required rather than defaulting to everyone: drawing from
    users the model trained on reports a number that has read the answers.
    Pass ``integrity.benchmark_users()``.
    """
    df = cf_mod.load_ratings()
    known = pl.Series("movieId", list(item_of_movielens.keys()), dtype=pl.Int32)
    df = df.filter(pl.col("movieId").is_in(known.implode()))

    # Sorted because `group_by` does not keep order: a seeded draw over an
    # unordered list picks different users every run.
    df = df.filter(pl.col("userId").is_in(pl.Series(np.asarray(eligible, dtype=np.int32)).implode()))
    counts = df.group_by("userId").len().filter(pl.col("len") >= min_history).sort("userId")
    rng = np.random.default_rng(seed)
    candidates = counts["userId"].to_numpy()
    if candidates.size == 0:
        return {}
    picked = rng.choice(candidates, size=min(n_users, candidates.size), replace=False)

    sub = df.filter(pl.col("userId").is_in(pl.Series(picked.astype(np.int32)).implode()))
    out: dict[int, dict[int, float]] = {}
    for uid, movie, rating in zip(
        sub["userId"].to_list(), sub["movieId"].to_list(), sub["rating"].to_list(), strict=True
    ):
        item = item_of_movielens.get(int(movie))
        if item is not None:
            out.setdefault(int(uid), {})[item] = float(rating)
    # Sorted too: users consume one shared generator in turn, so order decides
    # each user's split, and it should not rest on the ratings file's order.
    return {u: out[u] for u in sorted(out) if len(out[u]) >= min_history}


def _split(history: dict[int, float], eval_fraction: float, rng) -> tuple[dict, dict]:
    items = np.array(sorted(history.keys()))
    rng.shuffle(items)
    cut = int(len(items) * (1.0 - eval_fraction))
    known = {int(i): history[int(i)] for i in items[:cut]}
    held = {int(i): history[int(i)] for i in items[cut:]}
    return known, held


def _run_elicitation(
    fs: FeatureSpace,
    meta: dict[int, dict],
    known: dict[int, float],
    cfg: SimConfig,
    pool: np.ndarray,
    rng: np.random.Generator,
    seeds: list[int],
) -> list[tuple[int, float]]:
    """Play the two-phase elicitation against a simulated user.

    Returns the answered (item_id, reward) pairs. Questions about titles the
    user never rated are treated as "haven't seen it": they consume an ask but
    produce no label, exactly as they would with a real person.
    """
    answered: list[tuple[int, float]] = []
    asked: set[int] = set()
    cursor = 0

    while cursor < len(seeds):
        item = seeds[cursor]
        cursor += 1
        if len(answered) >= cfg.budget or len(asked) >= cfg.max_asks:
            break
        asked.add(item)
        if item in known:
            answered.append((item, _rating_to_reward(known[item])))
        if len(answered) >= 3 and cursor >= cfg.seed_questions * 4:
            break

    while len(answered) < cfg.budget and len(asked) < cfg.max_asks:
        if len(answered) < 3:
            # Not enough signal for a posterior yet; keep widening coverage
            # from the precomputed seed ladder.
            batch = [i for i in seeds[cursor:] if i not in asked][: cfg.seed_questions]
            cursor += len(batch)
            if not batch:
                break
        else:
            ids = np.array([a[0] for a in answered])
            rewards = np.array([a[1] for a in answered])
            model = fit_taste(
                fs.vectors_for(ids), rewards, allow_rff=False, prior=_ACTIVE_PRIOR
            )
            batch = elicit.next_questions(
                model, fs, meta, asked, k=cfg.seed_questions, pool=pool,
                criterion=cfg.criterion, rng=rng,
            )
        if not batch:
            break
        for item in batch:
            if len(answered) >= cfg.budget or len(asked) >= cfg.max_asks:
                break
            asked.add(item)
            if item in known:
                answered.append((item, _rating_to_reward(known[item])))
    return answered


def _random_elicitation(
    known: dict[int, float], cfg: SimConfig, rng: np.random.Generator
) -> list[tuple[int, float]]:
    """Control condition: ask about random titles the user has seen."""
    items = np.array(sorted(known.keys()))
    rng.shuffle(items)
    return [(int(i), _rating_to_reward(known[int(i)])) for i in items[: cfg.budget]]


# --- scoring arms -----------------------------------------------------------


def _rank(scores: np.ndarray, keep: np.ndarray, fs: FeatureSpace, k: int) -> list[int]:
    """Top-k item ids from scores over the *whole* item space.

    Arms score every row and mask afterwards rather than slicing the
    candidate set first. Slicing looked cheaper and was not: the excluded set
    is a few dozen titles out of seventy thousand, so `fs.matrix[rows]` copied
    ~58MB per arm per user — 2,400 copies across a run — to avoid scoring
    thirty rows. Masking also lets `argpartition` replace a full sort.
    """
    masked = np.where(keep, scores, -np.inf)
    top = np.argpartition(-masked, min(k, len(masked) - 1))[:k]
    top = top[np.argsort(-masked[top])]
    return [int(fs.item_ids[o]) for o in top if np.isfinite(masked[o])]


def arm_popularity(fs, meta, keep, answered, k):
    return _rank(fs.columns.votes.astype(np.float64), keep, fs, k)


def arm_quality(fs, meta, keep, answered, k):
    return _rank(fs.columns.quality.astype(np.float64), keep, fs, k)


def arm_content_centroid(fs, meta, keep, answered, k):
    """Cosine to the mean of the user's liked items — the standard naive baseline.

    Deliberately not mean-centred; see `arm_weighted_knn`.
    """
    liked = [i for i, r in answered if r >= 0.7]
    if not liked:
        return arm_quality(fs, meta, keep, answered, k)
    centroid = fs.latent[fs.rows_for(liked)].mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-9
    return _rank(fs.latent @ centroid, keep, fs, k)


def arm_weighted_knn(fs, meta, keep, answered, k, neighbours: int = 30):
    """Rating-weighted item kNN over the fused space.

    Deliberately *not* mean-centred. Centring is the textbook move and it is
    wrong for this data: people rate what they expected to like, so the
    verdicts sit in a narrow band near the top and centring converts "liked
    slightly less" into "disliked". It took this baseline from competitive to
    NDCG@10 0.007 — worse than randomly ordering a plausible shortlist.
    """
    if not answered:
        return arm_quality(fs, meta, keep, answered, k)
    ids = np.array([a[0] for a in answered])
    rewards = np.array([a[1] for a in answered])
    sims = fs.latent @ fs.latent[fs.rows_for(ids)].T
    top = min(neighbours, sims.shape[1])
    idx = np.argpartition(-sims, top - 1, axis=1)[:, :top]
    rows = np.arange(sims.shape[0])[:, None]
    s = np.maximum(sims[rows, idx], 0.0)
    # Accumulated, not averaged. Dividing by the similarity mass is the
    # textbook form and it throws away the only signal left once the rewards
    # are a narrow band: how strongly this title resembles things the person
    # liked at all. Normalised, every candidate scored ~0.78 and the ranking
    # was arbitrary — NDCG@10 0.003.
    scores = (s * rewards[idx]).sum(axis=1)
    return _rank(scores, keep, fs, k)


def arm_ridge(fs, meta, keep, answered, k):
    """Plain ridge regression on the same features: the model minus its Bayes.

    Given the same sampled negatives, so the comparison isolates the Bayesian
    treatment rather than re-measuring the contrast those negatives supply.
    """
    from sklearn.linear_model import RidgeCV

    if len(answered) < 3:
        return arm_quality(fs, meta, keep, answered, k)
    ids = np.array([a[0] for a in answered])
    rewards = np.array([a[1] for a in answered])
    X, y, w = _with_negatives(fs, ids, rewards)
    m = RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0)).fit(X, y, sample_weight=w)
    return _rank(m.predict(fs.matrix), keep, fs, k)


# Overridable so the count can be validated on held-out users rather than
# tuned on one person's history, where selecting it in-sample proves nothing.
NEGATIVE_SAMPLES = int(os.environ.get("ENTERTAINER_NEGATIVE_SAMPLES", "1000"))


def _with_negatives(fs, ids, rewards, seed=0):
    """Append sampled unrated titles as weak negatives.

    Mirrors what the engine does at serving time. Without this the model has
    no example of "not for me" and, because people rate things they expected
    to like, almost no variance to learn from either — see engine.py.
    """
    from ..engine import NEGATIVE_REWARD, NEGATIVE_WEIGHT

    rng = np.random.default_rng(seed)
    known = set(int(i) for i in ids)
    count = min(NEGATIVE_SAMPLES, len(fs.item_ids))
    rows = rng.choice(len(fs.item_ids), size=count, replace=False)
    rows = np.array([r for r in rows if int(fs.item_ids[r]) not in known], dtype=np.int64)

    X = np.vstack([fs.vectors_for(ids), fs.matrix[rows]])
    y = np.concatenate([rewards, np.full(rows.size, NEGATIVE_REWARD)])
    w = np.concatenate([np.ones(len(rewards)), np.full(rows.size, NEGATIVE_WEIGHT)])
    return X, y, w


def _taste_arm(fs, meta, keep, answered, k, prior, negatives=True):
    if len(answered) < 3:
        return arm_quality(fs, meta, keep, answered, k)
    ids = np.array([a[0] for a in answered])
    rewards = np.array([a[1] for a in answered])
    if negatives:
        X, y, w = _with_negatives(fs, ids, rewards)
    else:
        X, y, w = fs.vectors_for(ids), rewards, None
    model = fit_taste(
        X, y, sample_weight=w, prior=prior, capacity_obs=len(rewards)
    )
    mean = model.predict(fs.matrix, with_std=False)
    return _rank(mean, keep, fs, k)


def arm_taste_flat(fs, meta, keep, answered, k):
    """The engine with an isotropic prior: no population knowledge at all."""
    return _taste_arm(fs, meta, keep, answered, k, prior=None)


def arm_taste_no_negatives(fs, meta, keep, answered, k):
    """The engine without sampled negatives: positives-only regression."""
    return _taste_arm(fs, meta, keep, answered, k, prior=_ACTIVE_PRIOR, negatives=False)


def arm_taste(fs, meta, keep, answered, k):
    """The engine: evidence-tuned Bayesian posterior over a population prior.

    The prior is injected by ``run`` rather than loaded here, so that the
    replay can guarantee it was fitted without the simulated user's own
    opinions in it.
    """
    return _taste_arm(fs, meta, keep, answered, k, prior=_ACTIVE_PRIOR)


# Set by ``run``; module-level so the arm signature stays uniform.
_ACTIVE_PRIOR = None


ARMS = {
    "popularity": arm_popularity,
    "quality-prior": arm_quality,
    "content-centroid": arm_content_centroid,
    "weighted-kNN": arm_weighted_knn,
    "ridge": arm_ridge,
    "entertainer-flat-prior": arm_taste_flat,
    "entertainer-no-negatives": arm_taste_no_negatives,
    "entertainer": arm_taste,
}


def run(
    fs: FeatureSpace,
    meta: dict[int, dict],
    histories: dict[int, dict[int, float]],
    cfg: SimConfig,
    arms: Sequence[str] = tuple(ARMS),
    elicitation: str = "v-optimal",
    prior=None,
) -> dict[str, ArmResult]:
    """``elicitation``: v-optimal | d-optimal | random."""
    global _ACTIVE_PRIOR
    _ACTIVE_PRIOR = prior
    if elicitation in ("v-optimal", "d-optimal"):
        cfg = SimConfig(**{**cfg.__dict__, "criterion": elicitation})
    rng = np.random.default_rng(cfg.seed)
    results = {name: ArmResult(name) for name in arms}

    popularity = {
        int(fs.item_ids[r]): int((meta.get(int(fs.item_ids[r])) or {}).get("imdb_votes") or 0)
        for r in range(len(fs.item_ids))
    }
    pool = elicit.recognisable_pool(fs, meta)
    console.print(f"[dim]elicitation pool: {pool.size:,} recognisable titles[/dim]")

    # The opening ladder is the same for everyone by construction — it depends
    # only on the catalogue — so computing it per simulated user was pure
    # waste. Built once, long enough that a user who says "haven't seen it" to
    # most of it still has questions left.
    seeds = elicit.seed_questions(fs, meta, k=cfg.max_asks, pool=pool)
    console.print(f"[dim]opening ladder: {len(seeds)} titles[/dim]")

    with Progress(
        TextColumn("[bold blue]simulating"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} users"),
        TimeElapsedColumn(),
    ) as bar:
        task = bar.add_task("users", total=len(histories))
        for _uid, history in histories.items():
            known, held = _split(history, cfg.eval_fraction, rng)
            if not held:
                bar.advance(task)
                continue

            if elicitation == "random":
                answered = _random_elicitation(known, cfg, rng)
            else:
                answered = _run_elicitation(fs, meta, known, cfg, pool, rng, seeds)
            if len(answered) < 3:
                bar.advance(task)
                continue

            told = {i for i, _ in answered}
            # Candidates: everything except what the engine was told about.
            # The user's held-out ratings are in here, along with hundreds of
            # thousands of titles they never rated, which is the realistic
            # setting: the haystack is the whole catalogue.
            cand_mask = np.ones(len(fs.item_ids), dtype=bool)
            for i in told:
                cand_mask[fs.index[i]] = False

            relevance = {i: max(0.0, r - RELEVANT_AT + 1.0) for i, r in held.items() if r >= RELEVANT_AT}
            positives = set(relevance)
            if not positives:
                bar.advance(task)
                continue

            vectors = {}
            for name in arms:
                ranked = ARMS[name](fs, meta, cand_mask, answered, cfg.top_k)
                if not vectors:
                    vectors = {i: fs.latent[fs.index[i]] for i in ranked}
                else:
                    vectors.update({i: fs.latent[fs.index[i]] for i in ranked})
                results[name].per_user.append(
                    {
                        "ndcg@10": metrics.ndcg_at_k(ranked, relevance, cfg.top_k),
                        "precision@10": metrics.precision_at_k(ranked, positives, cfg.top_k),
                        "recall@10": metrics.recall_at_k(ranked, positives, cfg.top_k),
                        "map@10": metrics.average_precision(ranked, positives, cfg.top_k),
                        "mrr@10": metrics.reciprocal_rank(ranked, positives, cfg.top_k),
                        "novelty": metrics.novelty(ranked, popularity, cfg.top_k),
                        "diversity": metrics.intra_list_diversity(ranked, vectors, cfg.top_k),
                        "serendipity": metrics.serendipity(ranked, positives, popularity, cfg.top_k),
                        "answers": float(len(answered)),
                    }
                )
            bar.advance(task)
    return results


def paired_bootstrap(
    a: ArmResult, b: ArmResult, metric: str = "ndcg@10", iters: int = 10_000, seed: int = 0
) -> tuple[float, float]:
    """Paired bootstrap of (mean difference, P(a <= b)).

    Paired because both arms saw the same users with the same answers, so the
    user-to-user variance — which is enormous — cancels. An unpaired test here
    would hide almost any real effect.
    """
    n = min(len(a.per_user), len(b.per_user))
    if n == 0:
        return 0.0, 1.0
    da = np.array([r[metric] for r in a.per_user[:n]])
    db = np.array([r[metric] for r in b.per_user[:n]])
    diff = da - db
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(iters, n))
    boots = diff[idx].mean(axis=1)
    return float(diff.mean()), float((boots <= 0).mean())


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
