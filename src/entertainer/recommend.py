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

    def mask(self, fs: FeatureSpace) -> np.ndarray:
        """Boolean keep-mask over the item space.

        Vectorised over the precomputed metadata columns. The dict-walking
        version cost roughly a second per command at catalogue scale, which
        was most of the latency a user would have felt on `ent recs`.

        Unknown values fail a constraint rather than passing it: if a title's
        runtime is not recorded, "under two hours" cannot be honoured for it,
        and a hard constraint must never be quietly relaxed.
        """
        cols = fs.columns
        if cols is None:  # pragma: no cover - only for hand-built spaces
            raise RuntimeError("feature space has no metadata columns")

        keep = cols.known.copy()

        if self.exclude:
            excluded = np.fromiter(self.exclude, dtype=np.int64, count=len(self.exclude))
            keep &= ~np.isin(fs.item_ids, excluded)
        if self.languages:
            codes = cols.codes_for(self.languages)
            keep &= np.isin(cols.language_code, codes) if codes.size else np.False_
        if self.kind == "tv":
            keep &= cols.is_tv
        elif self.kind == "movie":
            keep &= ~cols.is_tv
        if self.min_year:
            keep &= cols.year >= self.min_year
        if self.max_year:
            keep &= (cols.year <= self.max_year) & (cols.year > 0)
        if self.max_runtime:
            keep &= (cols.runtime <= self.max_runtime) & (cols.runtime > 0)
        if self.min_runtime:
            keep &= cols.runtime >= self.min_runtime
        if self.min_votes:
            keep &= cols.votes >= self.min_votes
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


def _predictive_std(model: TasteModel, phi: np.ndarray) -> np.ndarray:
    """Predictive standard deviation for pre-mapped rows.

    Epistemic (parameter) variance plus aleatoric (noise) variance, computed
    through a Cholesky factor so the quadratic form is a single triangular
    solve rather than an explicit f x f sandwich per row.
    """
    chol = np.linalg.cholesky(model.cov + 1e-10 * np.eye(model.cov.shape[0]))
    projected = phi @ chol
    var = np.einsum("ij,ij->i", projected, projected) + 1.0 / model.beta
    return np.sqrt(np.maximum(var, 1e-12))


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
    mood: np.ndarray | None = None,
    mood_weight: float = 0.6,
) -> list[Recommendation]:
    rng = rng or np.random.default_rng()
    filters = filters or Filters()

    idx = np.flatnonzero(filters.mask(fs))
    if idx.size == 0:
        return []

    X = fs.matrix[idx]
    latent_all = fs.latent[idx]
    phi = model.feature_map(X)

    # Posterior mean is a single matrix-vector product, so it is affordable
    # across the whole catalogue. Predictive *variance* is a quadratic form
    # and costs f times as much — at 270k candidates and a lifted feature map
    # that is the difference between a recommendation appearing instantly and
    # taking fifteen seconds. It is therefore computed only for the shortlist,
    # which is the only place it changes any decision.
    mean_all = phi @ model.mean + model.y_mean

    if strategy == "mean":
        scores = mean_all
    elif strategy == "ucb":
        scores = mean_all  # provisional; corrected on the shortlist below
    else:
        w = model.sample_weights(rng, 1, temperature=max(explore, 0.0))[0]
        scores = phi @ w + model.y_mean

    if mood is not None and mood_weight > 0.0:
        # A free-text mood is a second opinion, not an override. It is blended
        # on a standardised scale so its influence is interpretable: at
        # weight 1.0 a title one standard deviation more on-mood outranks one
        # a standard deviation more to this person's taste. Filters stay hard;
        # this is deliberately soft, because "something slow tonight" is a
        # preference, not a constraint.
        affinity = latent_all @ mood
        affinity = (affinity - affinity.mean()) / (affinity.std() + 1e-9)
        scores = scores + mood_weight * float(np.std(scores)) * affinity

    if novelty > 0.0:
        # Push away from the canon. Scored on log votes rather than a hard
        # popularity cut, so a widely-loved film is discouraged rather than
        # banned — the user asked for the road less travelled, not for the
        # obscure at any cost.
        exposure = np.log1p(fs.columns.votes[idx].astype(np.float64))
        exposure = (exposure - exposure.mean()) / (exposure.std() + 1e-9)
        scores = scores - novelty * float(np.std(scores)) * exposure

    # Shortlist generously, then refine. The union with the top of the mean
    # ranking guarantees the greedy picks are always present, which is what
    # the explore/exploit marker is measured against.
    pool_size = min(max(shortlist, k * 20), idx.size)
    top = np.argsort(-scores)[:pool_size]
    top = np.union1d(top, np.argsort(-mean_all)[: max(k * 4, 64)])

    std_top = _predictive_std(model, phi[top])
    if strategy == "ucb":
        adjusted = mean_all[top] + max(explore, 0.0) * std_top
        reorder = np.argsort(-adjusted)
        top = top[reorder]
        std_top = std_top[reorder]
        scores = scores.copy()
        scores[top] = adjusted[reorder]

    std_by_row = dict(zip(top.tolist(), std_top.tolist(), strict=True))
    order = top[np.argsort(-scores[top])]

    diversified = _mmr(order, scores, latent_all, k=k * 3, lambda_=mmr_lambda)
    final_rows = _language_quota(diversified, fs.item_ids[idx], meta, k, max_language_share)

    # First-order propensity under a softmax over the shortlist. The slate is
    # drawn without replacement, so this understates later positions; it is
    # recorded as an approximation and used only for relative weighting.
    temp = max(float(np.std(scores[order])), 1e-3)
    logits = (scores[order] - scores[order].max()) / temp
    probs = np.exp(logits)
    probs /= probs.sum()
    prop_by_row = dict(zip(order.tolist(), probs.tolist(), strict=True))

    # "Explored" = the model's own point estimate did not rank it top-k, so it
    # is here because of posterior uncertainty rather than confidence.
    exploit_rows = set(np.argsort(-mean_all)[:k].tolist())

    out: list[Recommendation] = []
    for pos, row in enumerate(final_rows):
        out.append(
            Recommendation(
                item_id=int(fs.item_ids[idx[row]]),
                score=float(scores[row]),
                mean=float(mean_all[row]),
                std=float(std_by_row.get(row, float("nan"))),
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
