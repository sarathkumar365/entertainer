"""Ranking metrics.

Deliberately includes more than accuracy. A recommender optimised purely for
NDCG converges on the safest, most popular, most already-known titles — it
scores well and is useless, because a recommendation you were always going to
find is worth nothing. Coverage, novelty and intra-list diversity are reported
alongside so that a gain in accuracy bought by collapsing onto the canon is
visible rather than hidden.
"""

from __future__ import annotations

import math

import numpy as np


def dcg(relevances: np.ndarray) -> float:
    if relevances.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, relevances.size + 2))
    return float((relevances * discounts).sum())


def ndcg_at_k(ranked_ids: list[int], relevance: dict[int, float], k: int = 10) -> float:
    gains = np.array([relevance.get(i, 0.0) for i in ranked_ids[:k]], dtype=np.float64)
    ideal = np.array(sorted(relevance.values(), reverse=True)[:k], dtype=np.float64)
    denom = dcg(ideal)
    return dcg(gains) / denom if denom > 0 else 0.0


def precision_at_k(ranked_ids: list[int], positives: set[int], k: int = 10) -> float:
    if k == 0:
        return 0.0
    return len(set(ranked_ids[:k]) & positives) / k


def recall_at_k(ranked_ids: list[int], positives: set[int], k: int = 10) -> float:
    if not positives:
        return 0.0
    return len(set(ranked_ids[:k]) & positives) / len(positives)


def average_precision(ranked_ids: list[int], positives: set[int], k: int = 10) -> float:
    if not positives:
        return 0.0
    hits, total = 0, 0.0
    for rank, item in enumerate(ranked_ids[:k], start=1):
        if item in positives:
            hits += 1
            total += hits / rank
    return total / min(len(positives), k)


def reciprocal_rank(ranked_ids: list[int], positives: set[int], k: int = 10) -> float:
    for rank, item in enumerate(ranked_ids[:k], start=1):
        if item in positives:
            return 1.0 / rank
    return 0.0


def novelty(ranked_ids: list[int], popularity: dict[int, int], k: int = 10) -> float:
    """Mean self-information of the slate: higher means less obvious picks."""
    total = sum(popularity.values()) or 1
    vals = []
    for item in ranked_ids[:k]:
        p = (popularity.get(item, 0) + 1) / total
        vals.append(-math.log2(p))
    return float(np.mean(vals)) if vals else 0.0


def intra_list_diversity(ranked_ids: list[int], vectors: dict[int, np.ndarray], k: int = 10) -> float:
    """1 - mean pairwise cosine similarity within the slate."""
    vecs = [vectors[i] for i in ranked_ids[:k] if i in vectors]
    if len(vecs) < 2:
        return 0.0
    M = np.vstack(vecs)
    sims = M @ M.T
    n = len(vecs)
    off = (sims.sum() - np.trace(sims)) / (n * (n - 1))
    return float(1.0 - off)


def serendipity(
    ranked_ids: list[int],
    positives: set[int],
    popularity: dict[int, int],
    k: int = 10,
    popular_cutoff: int = 200,
) -> float:
    """Relevant hits that a popularity baseline would not have surfaced."""
    top_popular = {
        i for i, _ in sorted(popularity.items(), key=lambda kv: -kv[1])[:popular_cutoff]
    }
    hits = [i for i in ranked_ids[:k] if i in positives and i not in top_popular]
    return len(hits) / k if k else 0.0


def summarise(rows: list[dict]) -> dict[str, float]:
    """Mean and standard error over per-user metric dicts."""
    if not rows:
        return {}
    out: dict[str, float] = {}
    keys = rows[0].keys()
    n = len(rows)
    for key in keys:
        vals = np.array([r[key] for r in rows], dtype=np.float64)
        out[key] = float(vals.mean())
        out[f"{key}_se"] = float(vals.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    return out
