"""Cold start: work out someone's taste from nothing, in as few questions as possible.

On day one the posterior is the prior and every title looks equally good, so
the engine cannot recommend its way out of the problem — it has to ask. The
cost of asking is the user's patience, which is the scarcest resource in the
whole system. So the question is not "which titles are good" but "which titles,
once answered, most reduce uncertainty about this person".

Two phases, following the shape that the cold-start elicitation literature has
converged on.

**Phase one — a diverse seed.** Before any answer exists there is nothing to
adapt to, so the goal is coverage: a set of titles that spans the latent space
as widely as possible. Selected by greedy maximisation of the log-determinant
of a quality-weighted similarity kernel, which is the standard MAP
approximation for a determinantal point process. A DPP is the right object
here because it models *repulsion* — it will not hand back six acclaimed
American prestige dramas, because once it has taken one the others become
redundant to it.

**Phase two — adaptive questions.** Once answers start arriving the posterior
has shape, and the next question should be the one that sharpens it most. For
a Bayesian linear model the information gain from observing item x is exactly

    ½·log(1 + β·xᵀΣx)

which is monotone in the model's epistemic variance at x. So the optimal next
question is simply the title the model is least certain about — classical
sequential D-optimal design, in closed form, no approximation needed.

One practical constraint runs through both phases: there is no point asking
about a film the person has never heard of. A question about an unseen title
yields nothing, so the candidate pool is restricted by recognisability, with
the bar set per-language so that a widely-seen Malayalam film is not excluded
by a threshold calibrated on Hollywood viewing figures.
"""

from __future__ import annotations

import math

import numpy as np

from ..config import PRIORITY_LANGUAGES
from ..models.features import FeatureSpace
from ..models.taste import TasteModel

# Recognisability floors, as a fraction of the most-voted title in that
# language. Relative rather than absolute, because vote counts differ by two
# orders of magnitude across industries.
RECOGNISABILITY_QUANTILE = 0.90


def recognisable_pool(
    fs: FeatureSpace,
    meta: dict[int, dict],
    languages: tuple[str, ...] = (),
    quantile: float = RECOGNISABILITY_QUANTILE,
    per_language_cap: int | None = None,
) -> np.ndarray:
    """Row indices of titles a person plausibly has an opinion about."""
    wanted = set(languages) if languages else None
    by_lang: dict[str, list[tuple[float, int]]] = {}
    for row, iid in enumerate(fs.item_ids.tolist()):
        r = meta.get(int(iid))
        if not r:
            continue
        lang = r.get("language") or "xx"
        if wanted and lang not in wanted:
            continue
        votes = float(r.get("imdb_votes") or 0)
        if votes <= 0:
            continue
        by_lang.setdefault(lang, []).append((votes, row))

    keep: list[int] = []
    for lang, entries in by_lang.items():
        if len(entries) < 12:
            continue
        votes = np.array([e[0] for e in entries])
        floor = float(np.quantile(votes, quantile))
        picked = [row for v, row in entries if v >= floor]
        # Languages the user named get more room in the question budget.
        cap = per_language_cap or (1200 if lang in PRIORITY_LANGUAGES else 400)
        picked.sort(key=lambda row: -(meta[int(fs.item_ids[row])].get("imdb_votes") or 0))
        keep.extend(picked[:cap])
    return np.array(sorted(set(keep)), dtype=np.int64)


def greedy_dpp(
    vectors: np.ndarray,
    quality: np.ndarray,
    k: int,
    epsilon: float = 1e-9,
) -> list[int]:
    """Greedy MAP inference for a k-DPP over a quality-weighted cosine kernel.

    Uses the incremental Cholesky formulation, which keeps the whole thing
    O(k²n) rather than the O(k n³) of recomputing determinants. ``quality``
    scales each item's prior mass, so the process prefers well-regarded titles
    among otherwise equally diverse options.
    """
    n = vectors.shape[0]
    if n == 0:
        return []
    k = min(k, n)
    # d2[i] = squared "remaining volume" contributed by item i.
    d2 = quality.astype(np.float64) ** 2
    selected: list[int] = []
    cis = np.zeros((k, n), dtype=np.float64)

    j = int(np.argmax(d2))
    selected.append(j)
    for it in range(1, k):
        # Kernel row for the newly selected item, on the fly (n x d dot).
        sims = vectors @ vectors[j]
        L_j = quality * quality[j] * sims
        ci_opt = cis[:it, j]
        di_opt = math.sqrt(max(d2[j], epsilon))
        eis = (L_j - ci_opt @ cis[:it]) / di_opt
        cis[it - 1] = eis
        d2 = d2 - eis**2
        d2[selected] = -np.inf
        j = int(np.argmax(d2))
        if d2[j] < epsilon:
            break
        selected.append(j)
    return selected


def seed_questions(
    fs: FeatureSpace,
    meta: dict[int, dict],
    k: int = 24,
    languages: tuple[str, ...] = (),
    pool: np.ndarray | None = None,
) -> list[int]:
    """Phase one: a maximally spread-out opening set. Returns item_ids."""
    pool = recognisable_pool(fs, meta, languages) if pool is None else pool
    if pool.size == 0:
        return []
    vecs = fs.latent[pool]
    quality = np.array(
        [max(float((meta.get(int(fs.item_ids[r])) or {}).get("quality") or 0.5), 0.05) for r in pool]
    )
    chosen = greedy_dpp(vecs, quality, k=k)
    return [int(fs.item_ids[pool[c]]) for c in chosen]


def information_gain(model: TasteModel, X: np.ndarray) -> np.ndarray:
    """½·log(1 + β·xᵀΣx) for each row — exact for a Bayesian linear model."""
    phi = model.feature_map(X)
    epistemic = np.einsum("ij,jk,ik->i", phi, model.cov, phi)
    return 0.5 * np.log1p(model.beta * np.maximum(epistemic, 0.0))


def next_questions(
    model: TasteModel,
    fs: FeatureSpace,
    meta: dict[int, dict],
    asked: set[int],
    k: int = 8,
    languages: tuple[str, ...] = (),
    pool: np.ndarray | None = None,
    redundancy_lambda: float = 0.55,
) -> list[int]:
    """Phase two: the k most informative next questions, jointly chosen.

    Picking the top-k by information gain independently is a classic mistake:
    the most uncertain titles tend to be uncertain *for the same reason*, so
    answering one answers all of them. Greedy selection with a redundancy
    penalty against already-chosen questions approximates the batch-optimal
    design at negligible cost.
    """
    pool = recognisable_pool(fs, meta, languages) if pool is None else pool
    if pool.size == 0:
        return []
    keep = np.array([int(fs.item_ids[r]) not in asked for r in pool], dtype=bool)
    pool = pool[keep]
    if pool.size == 0:
        return []

    gains = information_gain(model, fs.matrix[pool])
    vecs = fs.latent[pool]

    chosen: list[int] = []
    redundancy = np.zeros(pool.size, dtype=np.float64)
    for _ in range(min(k, pool.size)):
        objective = gains - redundancy_lambda * redundancy
        if chosen:
            objective[np.array(chosen)] = -np.inf
        pick = int(np.argmax(objective))
        chosen.append(pick)
        redundancy = np.maximum(redundancy, np.abs(vecs @ vecs[pick]))
    return [int(fs.item_ids[pool[c]]) for c in chosen]
