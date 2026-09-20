"""Fuse the content and collaborative towers into one latent item space.

This is where the central design claim of the project lives.

The user supplies only titles and a like/dislike. Nothing about *why*. So the
representation the preference model learns over has to already contain the
axes along which "why" could vary — tone, pace, moral posture, formal
ambition, the particular texture of an actor's presence — without anyone
having named them.

Two independent sources provide those axes:

* the **content tower** embeds a prose card per title, so it carries whatever
  a multilingual sentence encoder can read off a synopsis and a cast list;
* the **collaborative tower** carries what no synopsis states — that a certain
  kind of viewer reliably crosses between two films that share no surface
  feature at all.

The two are concatenated and rotated by PCA into a single orthogonal basis.
The rotation matters: it decorrelates the axes, which is what lets a linear
preference model attribute credit cleanly, and it is what makes the learned
taste vector interpretable after the fact (see ``discover.py``).

Collaborative factors only exist for the ~87k titles MovieLens covers. For
everything else — which is most of the Tamil, Malayalam and Korean catalogue,
and every 2025 release — the factors are *imputed* from content by a ridge
map fitted on the overlap. Imputed factors are deliberately left shrunk
towards the mean rather than variance-inflated, and a per-item confidence
scalar is carried into the space so the downstream model can learn how much
to trust that block. Faking confidence the model does not have would be the
easy option and the wrong one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rich.console import Console
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from ..config import FUSED_DIM, PATHS

console = Console()

_RIDGE_ALPHAS = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0)


@dataclass
class FusionArtifacts:
    item_ids: np.ndarray        # (N,) catalogue item_id
    space: np.ndarray           # (N, D) fused latent coordinates, unit-normalised
    components: np.ndarray      # (D, F) PCA basis, for post-hoc axis naming
    block_sizes: tuple[int, int]
    cf_r2: float                # cross-validated quality of the content->CF map
    cf_coverage: float          # fraction of items with genuine CF factors


def _fit_cf_map(
    content_overlap: np.ndarray, cf_overlap: np.ndarray, seed: int = 0
) -> tuple[Ridge, float]:
    """Fit content -> CF-factor regression, choosing alpha by 5-fold CV R^2."""
    best_alpha, best_r2 = _RIDGE_ALPHAS[0], -np.inf
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for alpha in _RIDGE_ALPHAS:
        scores = []
        for tr, te in kf.split(content_overlap):
            m = Ridge(alpha=alpha).fit(content_overlap[tr], cf_overlap[tr])
            pred = m.predict(content_overlap[te])
            ss_res = float(((cf_overlap[te] - pred) ** 2).sum())
            ss_tot = float(((cf_overlap[te] - cf_overlap[tr].mean(0)) ** 2).sum())
            scores.append(1.0 - ss_res / max(ss_tot, 1e-9))
        mean_r2 = float(np.mean(scores))
        if mean_r2 > best_r2:
            best_alpha, best_r2 = alpha, mean_r2
    model = Ridge(alpha=best_alpha).fit(content_overlap, cf_overlap)
    console.print(f"[dim]content->CF ridge: alpha={best_alpha}, cv R^2={best_r2:.3f}[/dim]")
    return model, best_r2


def build(
    content_ids: np.ndarray,
    content: np.ndarray,
    cf_item_ids: np.ndarray,
    cf_factors: np.ndarray,
    movielens_of_item: dict[int, int],
    dim: int = FUSED_DIM,
    seed: int = 0,
) -> FusionArtifacts:
    """Construct the fused space.

    ``movielens_of_item`` maps catalogue item_id -> MovieLens movieId for the
    subset that has one.
    """
    n = len(content_ids)
    cf_pos = {int(m): i for i, m in enumerate(cf_item_ids.tolist())}

    # Align CF rows to catalogue rows where possible.
    cf_row = np.full(n, -1, dtype=np.int64)
    for i, item_id in enumerate(content_ids.tolist()):
        ml = movielens_of_item.get(int(item_id))
        if ml is not None and int(ml) in cf_pos:
            cf_row[i] = cf_pos[int(ml)]
    has_cf = cf_row >= 0
    coverage = float(has_cf.mean())
    console.print(f"[dim]genuine CF factors for {has_cf.sum():,}/{n:,} titles "
                  f"({coverage:.1%})[/dim]")

    cf_dim = cf_factors.shape[1]
    cf_block = np.zeros((n, cf_dim), dtype=np.float32)
    cf_block[has_cf] = cf_factors[cf_row[has_cf]]

    # Standardise the CF block on the observed rows only, so the imputation
    # target is well conditioned.
    cf_mean = cf_block[has_cf].mean(0)
    cf_std = cf_block[has_cf].std(0) + 1e-6
    cf_obs = (cf_block[has_cf] - cf_mean) / cf_std

    ridge, cf_r2 = _fit_cf_map(content[has_cf], cf_obs, seed=seed)
    cf_full = np.empty((n, cf_dim), dtype=np.float32)
    cf_full[has_cf] = cf_obs
    if (~has_cf).any():
        cf_full[~has_cf] = ridge.predict(content[~has_cf]).astype(np.float32)

    # Confidence in the CF block: 1 where measured, R^2-attenuated where
    # imputed. Carried as a feature rather than used to rescale, so the
    # preference model can decide what to do with it.
    confidence = np.where(has_cf, 1.0, max(cf_r2, 0.0)).astype(np.float32)[:, None]

    # Scale blocks to comparable energy before concatenation, otherwise PCA
    # simply reports whichever block happens to have larger variance.
    c_scaled = content / (np.linalg.norm(content, axis=1, keepdims=True) + 1e-9)
    f_scaled = cf_full / (np.linalg.norm(cf_full, axis=1, keepdims=True) + 1e-9)
    joint = np.hstack([c_scaled, f_scaled, confidence]).astype(np.float32)

    k = min(dim, joint.shape[1], joint.shape[0])
    pca = PCA(n_components=k, svd_solver="randomized", random_state=seed, whiten=False)
    space = pca.fit_transform(joint).astype(np.float32)
    evr = float(pca.explained_variance_ratio_.sum())
    console.print(f"[dim]PCA {joint.shape[1]} -> {k} dims, {evr:.1%} variance retained[/dim]")

    space /= np.linalg.norm(space, axis=1, keepdims=True) + 1e-9

    return FusionArtifacts(
        item_ids=content_ids.astype(np.int32),
        space=space,
        components=pca.components_.astype(np.float32),
        block_sizes=(content.shape[1], cf_dim),
        cf_r2=cf_r2,
        cf_coverage=coverage,
    )


def save(art: FusionArtifacts) -> None:
    PATHS.ensure()
    np.savez_compressed(
        PATHS.embeddings / "fused.npz",
        item_ids=art.item_ids,
        space=art.space,
        components=art.components,
        block_sizes=np.array(art.block_sizes),
        cf_r2=np.array([art.cf_r2]),
        cf_coverage=np.array([art.cf_coverage]),
    )


def load() -> FusionArtifacts:
    z = np.load(PATHS.embeddings / "fused.npz")
    return FusionArtifacts(
        item_ids=z["item_ids"],
        space=z["space"],
        components=z["components"],
        block_sizes=tuple(int(x) for x in z["block_sizes"]),
        cf_r2=float(z["cf_r2"][0]),
        cf_coverage=float(z["cf_coverage"][0]),
    )


def exists() -> bool:
    return (PATHS.embeddings / "fused.npz").exists()
