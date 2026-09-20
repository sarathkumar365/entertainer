"""Turn a taste posterior into a slate of titles.

Three jobs, in order.

**Candidate filtering.** Hard constraints the user actually stated (language,
film vs series, runtime, era) are applied as filters, never as score
penalties. A hard constraint expressed as a soft penalty is a bug: it produces
a slate that mostly obeys the constraint, which is worse than useless when the
constraint was the point.

**Scoring by Thompson sampling.** One draw is taken from the posterior and
every candidate scored under it. This is what makes the system explore: on any
given evening the engine commits to one coherent hypothesis about the user and
plays it out, rather than hedging towards the safe middle of the distribution.
Over many evenings the hypotheses that keep being confirmed dominate.

**Diversification.** A slate of ten near-identical films is ten times less
informative than a spread, both for the user and for the model. Maximal
marginal relevance trades score against redundancy; a soft language quota
keeps a multilingual catalogue from collapsing to whichever language happens
to dominate the posterior this week.

Every slate is logged with the probability each item had of being shown, which
is what makes honest off-policy evaluation possible later. Without propensity
logging, "did the model get better" can only ever be answered by vibes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .models.features import FeatureSpace
from .models.taste import TasteModel


@dataclass
class Filters:
    languages: tuple[str, ...] = ()
    kind: str | None = None          # 'movie' | 'tv'
    min_year: int | None = None
    max_year: int | None = None
    max_runtime: int | None = None
    min_runtime: int | None = None
    min_votes: int | None = None
    exclude: frozenset[int] = frozenset()

    def mask(self, item_ids: np.ndarray, meta: dict[int, dict]) -> np.ndarray:
        keep = np.ones(len(item_ids), dtype=bool)
        langs = set(self.languages)
        for i, iid in enumerate(item_ids.tolist()):
            iid = int(iid)
            if iid in self.exclude:
                keep[i] = False
                continue
            r = meta.get(iid)
            if r is None:
                keep[i] = False
                continue
            if langs and (r.get("language") not in langs) or self.kind and r.get("kind") != self.kind or self.min_year and (r.get("year") or 0) < self.min_year or self.max_year and (r.get("year") or 9999) > self.max_year or self.max_runtime and (r.get("runtime") or 0) > self.max_runtime or self.min_runtime and (r.get("runtime") or 0) < self.min_runtime or self.min_votes and (r.get("imdb_votes") or 0) < self.min_votes:
                keep[i] = False
        return keep


@dataclass
class Recommendation:
    item_id: int
    score: float
    mean: float
    std: float
    propensity: float
    explored: bool
    position: int
    reasons: list[tuple[str, float]] = field(default_factory=list)


def _mmr(
    order: np.ndarray,
    scores: np.ndarray,
    latent: np.ndarray,
    k: int,
    lambda_: float,
) -> list[int]:
    """Maximal marginal relevance over the candidate pool.

    Operates on a pre-truncated shortlist because the pairwise similarity term
    is quadratic; there is no value in diversifying against candidates that
    were never going to be shown.
    """
    selected: list[int] = []
    pool = list(order)
    if not pool:
        return selected
    sims = np.zeros(len(pool), dtype=np.float32)
    pool_vecs = latent[pool]
    pool_scores = scores[pool]
    chosen_mask = np.zeros(len(pool), dtype=bool)

    for _ in range(min(k, len(pool))):
        objective = lambda_ * pool_scores - (1.0 - lambda_) * sims
        objective[chosen_mask] = -np.inf
        pick = int(np.argmax(objective))
        chosen_mask[pick] = True
        selected.append(pool[pick])
        new_sims = pool_vecs @ pool_vecs[pick]
        sims = np.maximum(sims, new_sims)
    return selected


def _language_quota(
    ranked: list[int],
    item_ids: np.ndarray,
    meta: dict[int, dict],
    k: int,
    max_share: float,
) -> list[int]:
    """Cap how much of a slate any single language may occupy."""
    cap = max(1, int(round(k * max_share)))
    out: list[int] = []
    used: dict[str, int] = {}
    overflow: list[int] = []
    for row in ranked:
        lang = (meta.get(int(item_ids[row])) or {}).get("language") or "xx"
        if used.get(lang, 0) >= cap:
            overflow.append(row)
            continue
        used[lang] = used.get(lang, 0) + 1
        out.append(row)
        if len(out) == k:
            return out
    out.extend(overflow[: k - len(out)])
    return out


def recommend(
    model: TasteModel,
    fs: FeatureSpace,
    meta: dict[int, dict],
    k: int = 10,
    filters: Filters | None = None,
    rng: np.random.Generator | None = None,
    strategy: str = "thompson",
    mmr_lambda: float = 0.72,
    shortlist: int = 400,
    max_language_share: float = 0.5,
    explore: float = 1.0,
    novelty: float = 0.0,
) -> list[Recommendation]:
    rng = rng or np.random.default_rng()
    filters = filters or Filters()

    mask = filters.mask(fs.item_ids, meta)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []

    X = fs.matrix[idx]
    mean, std = model.predict(X)

    if strategy == "mean":
        scores = mean
    elif strategy == "ucb":
        scores = model.ucb_scores(X, kappa=max(explore, 0.0))
    else:
        scores = model.thompson_scores(X, rng, temperature=max(explore, 0.0))

    if novelty > 0.0:
        # Push away from the canon. Scored on log votes rather than a hard
        # popularity cut, so a widely-loved film is discouraged rather than
        # banned — the user asked for the road less travelled, not for the
        # obscure at any cost.
        votes = np.array(
            [float((meta.get(int(i)) or {}).get("imdb_votes") or 0) for i in fs.item_ids[idx]]
        )
        exposure = np.log1p(votes)
        exposure = (exposure - exposure.mean()) / (exposure.std() + 1e-9)
        scores = scores - novelty * float(np.std(scores)) * exposure

    top = np.argsort(-scores)[: min(shortlist, idx.size)]
    latent = fs.latent[idx]
    diversified = _mmr(top, scores, latent, k=k * 3, lambda_=mmr_lambda)
    final_rows = _language_quota(diversified, fs.item_ids[idx], meta, k, max_language_share)

    # First-order propensity under a softmax over the shortlist. The slate is
    # drawn without replacement, so this understates later positions; it is
    # recorded as an approximation and used only for relative weighting.
    temp = max(float(np.std(scores[top])), 1e-3)
    logits = (scores[top] - scores[top].max()) / temp
    probs = np.exp(logits)
    probs /= probs.sum()
    prop_by_row = dict(zip(top.tolist(), probs.tolist(), strict=True))

    # "Explored" = the model's own point estimate did not rank it top-k, so it
    # is here because of posterior uncertainty rather than confidence.
    exploit_rows = set(np.argsort(-mean)[:k].tolist())

    out: list[Recommendation] = []
    for pos, row in enumerate(final_rows):
        out.append(
            Recommendation(
                item_id=int(fs.item_ids[idx[row]]),
                score=float(scores[row]),
                mean=float(mean[row]),
                std=float(std[row]),
                propensity=float(prop_by_row.get(row, 1.0 / max(idx.size, 1))),
                explored=row not in exploit_rows,
                position=pos,
            )
        )
    return out


def attach_reasons(
    recs: list[Recommendation],
    fs: FeatureSpace,
    liked_item_ids: list[int],
    liked_labels: list[str],
    top: int = 3,
) -> None:
    """Annotate each recommendation with the user's own titles it resembles."""
    from .models.discover import nearest_liked

    if not liked_item_ids:
        return
    liked_rows = fs.rows_for(liked_item_ids)
    liked_vecs = fs.latent[liked_rows]
    for rec in recs:
        vec = fs.latent[fs.index[rec.item_id]]
        rec.reasons = nearest_liked(vec, liked_vecs, liked_labels, top=top)
