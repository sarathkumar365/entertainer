"""Read the learned taste vector back out in words.

The user never says why they liked something. The model never asks. So the
only way anyone finds out *what was learned* is to invert the representation
after the fact — take the axes of the latent space the preference weights
actually fire on, look at which titles sit at each end, and work out what
distinguishes those two piles.

Two things are worth being precise about.

First, the axes are discovered, not chosen. Nothing in the pipeline says "one
dimension shall mean pacing". PCA over the fused content+collaborative space
produces whatever directions carry variance; the preference model then puts
weight on some of them. Which ones turn out to matter is an empirical result
about one person, not a design decision.

Second, the *labels* on those axes are produced here, at the end, purely so a
human can read them. Genre and keyword vocabulary is used as a describing
language, never as an input to scoring. This ordering is the whole point: tags
explain the model's findings, they do not constrain them.

Pole descriptions use the weighted log-odds ratio with an informative Dirichlet
prior (Monroe, Colaresi & Quinn, 2008), which is the standard fix for the
failure mode where naive frequency comparison just reports whatever is common
everywhere, and naive ratio comparison just reports whatever is rare.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .taste import TasteModel, taste_direction

# Terms that describe production circumstances rather than the experience of
# watching. They dominate keyword vocabularies and say nothing about taste.
_STOP_TERMS = {
    "based on novel or book", "based on true story", "woman director",
    "aftercreditsstinger", "duringcreditsstinger", "live action",
    "independent film", "sequel", "remake", "3d", "imax",
}


@dataclass
class Axis:
    index: int
    weight: float          # signed preference weight on this axis
    strength: float        # |weight| * axis spread, i.e. realised influence
    liked_pole: list[str]  # descriptive terms at the end the user prefers
    other_pole: list[str]
    liked_examples: list[str]
    other_examples: list[str]


def _terms_for(row: dict) -> list[str]:
    out: list[str] = []
    for g in row.get("genres") or []:
        if g:
            out.append(g.lower())
    for k in (row.get("keywords") or [])[:20]:
        if k and k.lower() not in _STOP_TERMS:
            out.append(k.lower())
    for d in (row.get("directors") or [])[:2]:
        if d:
            out.append(f"dir: {d}")
    lang = row.get("language")
    if lang:
        out.append(f"lang: {lang}")
    decade = row.get("year")
    if decade:
        out.append(f"{(int(decade) // 10) * 10}s")
    if row.get("kind") == "tv":
        out.append("series")
    runtime = row.get("runtime")
    if runtime:
        if runtime >= 150:
            out.append("long runtime")
        elif runtime <= 95:
            out.append("short runtime")
    return out


def _log_odds(
    pole: Counter, other: Counter, background: Counter, prior_strength: float = 500.0, top: int = 8
) -> list[str]:
    """Weighted log-odds with an informative Dirichlet prior."""
    total_bg = sum(background.values()) or 1
    n_a, n_b = sum(pole.values()), sum(other.values())
    scored: list[tuple[float, str]] = []
    for term in set(pole) | set(other):
        a0 = prior_strength * background[term] / total_bg
        if a0 <= 0:
            continue
        ya, yb = pole[term], other[term]
        if ya + yb < 3:
            continue
        odds_a = (ya + a0) / (n_a + prior_strength - ya - a0)
        odds_b = (yb + a0) / (n_b + prior_strength - yb - a0)
        delta = math.log(odds_a) - math.log(odds_b)
        var = 1.0 / (ya + a0) + 1.0 / (yb + a0)
        scored.append((delta / math.sqrt(var), term))
    scored.sort(reverse=True)
    return [t for _, t in scored[:top]]


def _background_counts(
    rows: dict[int, dict], sample: int = 40_000, seed: int = 0
) -> Counter:
    """How common each descriptive term is in the catalogue at large.

    Sampled rather than exhaustive. The log-odds prior only needs each term's
    relative frequency, which forty thousand titles estimate perfectly well,
    and counting all of them costs seconds on every invocation of a command
    that should feel instant.
    """
    keys = list(rows)
    if len(keys) > sample:
        rng = np.random.default_rng(seed)
        keys = [keys[i] for i in rng.choice(len(keys), size=sample, replace=False)]
    background = Counter()
    for key in keys:
        background.update(_terms_for(rows[key]))
    return background


def describe_axes(
    model: TasteModel,
    item_ids: np.ndarray,
    space: np.ndarray,
    rows: dict[int, dict],
    n_axes: int = 6,
    pole_size: int = 400,
    example_count: int = 4,
    background: Counter | None = None,
) -> list[Axis]:
    """Name the latent axes this person's preferences actually rest on."""
    w = taste_direction(model)
    n_latent = space.shape[1]
    if w.size < n_latent:
        return []
    # The feature vector is [latent axes | side features]; only the latent
    # block has a geometry that can be interpreted by looking at poles.
    w = w[:n_latent]

    spread = space.std(axis=0)
    strength = np.abs(w) * spread
    order = np.argsort(-strength)[:n_axes]

    background = background if background is not None else _background_counts(rows)

    axes: list[Axis] = []
    for idx in order:
        col = space[:, idx]
        rank = np.argsort(col)
        low_ids = item_ids[rank[:pole_size]]
        high_ids = item_ids[rank[-pole_size:]]

        low_terms, high_terms = Counter(), Counter()
        for i in low_ids.tolist():
            if i in rows:
                low_terms.update(_terms_for(rows[i]))
        for i in high_ids.tolist():
            if i in rows:
                high_terms.update(_terms_for(rows[i]))

        liked_high = w[idx] >= 0
        liked_ids = high_ids if liked_high else low_ids
        other_ids = low_ids if liked_high else high_ids
        liked_terms = high_terms if liked_high else low_terms
        other_terms = low_terms if liked_high else high_terms

        def examples(ids: np.ndarray) -> list[str]:
            cands = [rows[i] for i in ids.tolist() if i in rows]
            cands.sort(key=lambda r: -(r.get("imdb_votes") or 0))
            return [
                f"{c['title']}" + (f" ({c['year']})" if c.get("year") else "")
                for c in cands[:example_count]
            ]

        axes.append(
            Axis(
                index=int(idx),
                weight=float(w[idx]),
                strength=float(strength[idx]),
                liked_pole=_log_odds(liked_terms, other_terms, background),
                other_pole=_log_odds(other_terms, liked_terms, background),
                liked_examples=examples(liked_ids),
                other_examples=examples(other_ids),
            )
        )
    return axes


def nearest_liked(
    candidate_vec: np.ndarray,
    liked_vecs: np.ndarray,
    liked_labels: list[str],
    top: int = 3,
    min_sim: float = 0.15,
) -> list[tuple[str, float]]:
    """Which of the person's own liked titles this recommendation resembles.

    This is the explanation that actually lands. "Because it is 0.82 aligned
    with latent axis 14" means nothing to anybody; "because you loved these
    three, and this sits between them" is checkable by the reader.
    """
    if liked_vecs.size == 0:
        return []
    sims = liked_vecs @ candidate_vec
    order = np.argsort(-sims)[:top]
    return [(liked_labels[i], float(sims[i])) for i in order if sims[i] >= min_sim]


def taste_summary(model: TasteModel, axes: list[Axis]) -> list[str]:
    """A few plain sentences about what the model believes."""
    lines: list[str] = []
    for ax in axes:
        liked = ", ".join(ax.liked_pole[:5]) or "—"
        against = ", ".join(ax.other_pole[:4]) or "—"
        lines.append(
            f"axis {ax.index:>3} (influence {ax.strength:.3f}): pulls towards [{liked}] "
            f"and away from [{against}]"
        )
    return lines


def surface_preferences(model: TasteModel, fs) -> list[tuple[str, float]]:
    """The five named side features, with their learned signed weights.

    The 192 latent axes are only describable by their poles, but these five
    have stable human names — consensus quality, how widely seen, release
    recency, runtime, is a series — so they can be read directly.

    Returns an empty list when the model's feature vector does not line up
    with the names, which happens if a model is loaded against an item space
    of a different shape. Silently mislabelling weights would be worse than
    showing none.
    """
    from .features import SIDE_FEATURE_NAMES

    side = taste_direction(model)[fs.n_latent :]
    if side.size != len(SIDE_FEATURE_NAMES):
        return []
    return [(name, float(w)) for name, w in zip(SIDE_FEATURE_NAMES, side, strict=True)]
