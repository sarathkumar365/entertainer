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


def _allocate(languages: list[tuple[str, int]], k: int, cap: int) -> dict[str, int]:
    """Spread k questions over the available languages, round-robin under a cap.

    Round-robin rather than proportional. Proportional allocation reproduces
    the catalogue's imbalance, which is exactly the thing being corrected: the
    point of the opening set is to find out whether this person watches
    Malayalam films at all, and that question cannot be asked with 0.4 of a
    question.
    """
    quota: dict[str, int] = {lang: 0 for lang, _ in languages}
    remaining = k
    while remaining > 0:
        progressed = False
        for lang, available in languages:
            if remaining == 0:
                break
            if quota[lang] >= min(cap, available):
                continue
            quota[lang] += 1
            remaining -= 1
            progressed = True
        if not progressed:
            break
    return {lang: n for lang, n in quota.items() if n > 0}


def seed_questions(
    fs: FeatureSpace,
    meta: dict[int, dict],
    k: int = 24,
    languages: tuple[str, ...] = (),
    pool: np.ndarray | None = None,
    max_language_share: float = 0.4,
) -> list[int]:
    """Phase one: a maximally spread-out opening set. Returns item_ids.

    A determinantal point process spreads over the latent space, but the
    catalogue is not balanced — English titles outnumber Malayalam ones by an
    order of magnitude and carry most of the high-quality mass. Run
    unconstrained over the whole pool, the DPP opens with a solid run of
    Anglophone prestige cinema and learns nothing about the rest of someone's
    taste.

    Trimming that set afterwards does not help, because the non-English titles
    were never selected in the first place. So the budget is allocated across
    languages first, and a separate DPP runs inside each one. Diversity is
    still maximised, but within a stratum rather than across a skewed whole.
    """
    pool = recognisable_pool(fs, meta, languages) if pool is None else pool
    if pool.size == 0:
        return []

    by_lang: dict[str, list[int]] = {}
    for row in pool.tolist():
        lang = (meta.get(int(fs.item_ids[row])) or {}).get("language") or "xx"
        by_lang.setdefault(lang, []).append(row)

    # Largest pools first, so round-robin starts from the languages most
    # likely to be relevant and the tail still gets its turn.
    ordered = sorted(by_lang.items(), key=lambda kv: -len(kv[1]))
    cap = max(1, int(round(k * max_language_share)))
    quota = _allocate([(lang, len(rows)) for lang, rows in ordered], k, cap)

    picked: list[int] = []
    for lang, rows in ordered:
        want = quota.get(lang, 0)
        if want <= 0:
            continue
        idx = np.array(rows, dtype=np.int64)
        quality = np.array(
            [
                max(float((meta.get(int(fs.item_ids[r])) or {}).get("quality") or 0.5), 0.05)
                for r in idx
            ]
        )
        chosen = greedy_dpp(fs.latent[idx], quality, k=want)
        picked.extend(int(fs.item_ids[idx[c]]) for c in chosen)

    # Interleave languages so the opening questions alternate rather than
    # arriving in blocks, which reads as a fairer sample to the person
    # answering and produces better coverage if they stop early.
    groups: dict[str, list[int]] = {}
    for item in picked:
        groups.setdefault((meta.get(item) or {}).get("language") or "xx", []).append(item)
    out: list[int] = []
    while len(out) < min(k, len(picked)):
        added = False
        for lang in list(groups):
            if groups[lang]:
                out.append(groups[lang].pop(0))
                added = True
                if len(out) == k:
                    break
        if not added:
            break
    return out


def information_gain(model: TasteModel, X: np.ndarray) -> np.ndarray:
    """D-optimal criterion: the Shannon information a label at x yields.

    For a Bayesian linear model this is exactly ½·log(1 + beta·x'Sigma·x), i.e.
    monotone in the model's epistemic variance at x.

    Kept because it is the textbook answer and a useful comparison, but it is
    not the default — see ``variance_reduction``.
    """
    phi = model.feature_map(X)
    epistemic = np.einsum("ij,jk,ik->i", phi, model.cov, phi)
    return 0.5 * np.log1p(model.beta * np.maximum(epistemic, 0.0))


def variance_reduction(
    model: TasteModel, X: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    """V-optimal criterion: how much a label at x sharpens the whole catalogue.

    D-optimality above answers "which label tells me most about my
    parameters". That is not quite the question. The engine does not want
    well-determined parameters for their own sake; it wants to rank a specific
    catalogue well. The quantity that matches that goal is the reduction in
    *predictive* variance summed over the titles it will actually have to
    rank.

    For a Bayesian linear model the rank-one posterior update

        Sigma' = Sigma - beta·Sigma·x·x'·Sigma / (1 + beta·x'·Sigma·x)

    gives that reduction over a reference set Z in closed form:

        dV(x) = beta · x'·Sigma·M·Sigma·x / (1 + beta·x'·Sigma·x),   M = Z'Z/|Z|

    so it costs one precomputed f x f matrix and a single quadratic form per
    candidate — no more expensive than the D-optimal score it replaces.

    Which of the two actually produces better recommendations is an empirical
    question, not a theoretical one, and the synthetic fixtures are too clean
    to settle it: everything saturates past about twenty questions there. It
    is settled instead on held-out MovieLens users — run
    ``ent eval --elicitation d-optimal`` against ``v-optimal`` and compare.
    V-optimal is the default because it optimises the objective the system is
    actually judged on; see docs/RESULTS.md for the measurement.

    ``reference`` is a sample of the catalogue in feature space; sampling
    suffices, since M only needs the second-moment structure.
    """
    phi = model.feature_map(X)
    ref = model.feature_map(reference)

    cov = model.cov
    M = ref.T @ ref / max(len(ref), 1)
    A = cov @ M @ cov                      # f x f, precomputed once

    numer = np.einsum("ij,jk,ik->i", phi, A, phi)
    denom = 1.0 + model.beta * np.einsum("ij,jk,ik->i", phi, cov, phi)
    return model.beta * np.maximum(numer, 0.0) / np.maximum(denom, 1e-12)


def next_questions(
    model: TasteModel,
    fs: FeatureSpace,
    meta: dict[int, dict],
    asked: set[int],
    k: int = 8,
    languages: tuple[str, ...] = (),
    pool: np.ndarray | None = None,
    redundancy_lambda: float = 0.55,
    criterion: str = "v-optimal",
    reference_size: int = 4000,
    rng: np.random.Generator | None = None,
) -> list[int]:
    """Phase two: the k most useful next questions, jointly chosen.

    Picking the top-k by score independently is a classic mistake: the most
    informative titles tend to be informative *for the same reason*, so
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

    if criterion == "d-optimal":
        gains = information_gain(model, fs.matrix[pool])
    else:
        rng = rng or np.random.default_rng(0)
        n_ref = min(reference_size, len(fs.item_ids))
        ref_rows = rng.choice(len(fs.item_ids), size=n_ref, replace=False)
        gains = variance_reduction(model, fs.matrix[pool], fs.matrix[ref_rows])
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
