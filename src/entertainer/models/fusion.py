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

from ..config import CF_RANK, FUSED_DIM, PATHS
from ..errors import ModelNotReady

console = Console()

_RIDGE_ALPHAS = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0)


def fused_path():
    return PATHS.embeddings / "fused.npz"


@dataclass
class FusionArtifacts:
    item_ids: np.ndarray        # (N,) catalogue item_id
    space: np.ndarray           # (N, D) fused latent coordinates, unit-normalised
    components: np.ndarray      # (D, F) PCA basis, for post-hoc axis naming
    block_sizes: tuple[int, int]
    cf_r2: float                # cross-validated quality of the content->CF map
    cf_coverage: float          # fraction of items with genuine CF factors
    # Everything below exists so a title that was never in the catalogue can
    # be projected into the same space without rebuilding it. Without these
    # the space is a closed world, and a user naming a film too obscure to
    # have made the vote floor would simply be told no.
    pca_mean: np.ndarray | None = None
    ridge_coef: np.ndarray | None = None
    ridge_intercept: np.ndarray | None = None

    def project(self, content: np.ndarray) -> np.ndarray:
        """Map raw content embeddings into the fused space, out of sample.

        Mirrors ``build`` exactly: impute the collaborative block from
        content, normalise each block, append the imputation-confidence
        scalar, rotate by the stored PCA basis, renormalise. A new title has
        no genuine collaborative factors by definition, so its confidence is
        the map's cross-validated R^2 — the same value every imputed title in
        the catalogue carries.
        """
        if self.pca_mean is None or self.ridge_coef is None:
            raise ModelNotReady(
                "this fused space predates out-of-sample projection; "
                "run `ent data fuse` to rebuild it"
            )

        content = np.atleast_2d(np.asarray(content, dtype=np.float32))
        cf = content @ self.ridge_coef.T + self.ridge_intercept

        c_scaled = content / (np.linalg.norm(content, axis=1, keepdims=True) + 1e-9)
        f_scaled = cf / (np.linalg.norm(cf, axis=1, keepdims=True) + 1e-9)
        confidence = np.full((content.shape[0], 1), max(self.cf_r2, 0.0), dtype=np.float32)

        joint = np.hstack([c_scaled, f_scaled, confidence]).astype(np.float32)
        out = (joint - self.pca_mean) @ self.components.T
        return (out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)).astype(np.float32)


def _cv_r2_by_alpha(
    X: np.ndarray, Y: np.ndarray, alphas: tuple[float, ...], seed: int = 0
) -> np.ndarray:
    """Mean 5-fold R^2 for every alpha, one eigendecomposition per fold.

    Equivalent to fitting ``Ridge(alpha)`` per (alpha, fold) pair, but the
    fold's centred Gram matrix is decomposed once and every alpha is read off
    the same spectrum. RidgeCV's closed-form LOO was the cheaper option; it
    was passed over because ``cf_r2`` is not only a selection score but the
    confidence every imputed title carries into the fused space, and LOO
    would silently shift that value relative to earlier builds.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    scores = np.zeros((len(alphas), kf.get_n_splits()))
    for f, (tr, te) in enumerate(kf.split(X)):
        x_mean, y_mean = X[tr].mean(0), Y[tr].mean(0)
        Xc = X[tr] - x_mean
        evals, evecs = np.linalg.eigh(Xc.T @ Xc)
        proj = evecs.T @ (Xc.T @ (Y[tr] - y_mean))
        Xte = (X[te] - x_mean) @ evecs
        ss_tot = float(((Y[te] - y_mean) ** 2).sum())
        for k, alpha in enumerate(alphas):
            pred = Xte @ (proj / (evals + alpha)[:, None]) + y_mean
            ss_res = float(((Y[te] - pred) ** 2).sum())
            scores[k, f] = 1.0 - ss_res / max(ss_tot, 1e-9)
    return scores.mean(axis=1)


def _fit_cf_map(
    content_overlap: np.ndarray, cf_overlap: np.ndarray, seed: int = 0
) -> tuple[Ridge, float]:
    """Fit content -> CF-factor regression, choosing alpha by 5-fold CV R^2."""
    r2 = _cv_r2_by_alpha(content_overlap, cf_overlap, _RIDGE_ALPHAS, seed=seed)
    best = int(np.argmax(r2))
    best_alpha, best_r2 = _RIDGE_ALPHAS[best], float(r2[best])
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
    cf_rank: int = CF_RANK,
) -> FusionArtifacts:
    """Construct the fused space.

    ``movielens_of_item`` maps catalogue item_id -> MovieLens movieId for the
    subset that has one.
    """
    n = len(content_ids)
    cf_pos = {int(m): i for i, m in enumerate(cf_item_ids.tolist())}

    # Align CF rows to catalogue rows where possible.
    cf_row = np.fromiter(
        (cf_pos.get(movielens_of_item.get(int(i), -1), -1) for i in content_ids.tolist()),
        dtype=np.int64, count=n,
    )
    has_cf = cf_row >= 0
    coverage = float(has_cf.mean())
    console.print(f"[dim]genuine CF factors for {has_cf.sum():,}/{n:,} titles "
                  f"({coverage:.1%})[/dim]")

    observed = cf_factors[cf_row[has_cf]]

    # Truncate the factors to their leading principal components before
    # anything else. See config.CF_RANK for the measurements: the trailing
    # dimensions carry variance but almost no similarity structure, and they
    # are the part content cannot predict, so keeping them halves the
    # imputation accuracy for the majority of the catalogue while buying
    # under two per cent of the geometry.
    cf_mean = observed.mean(0)
    rank = min(cf_rank, observed.shape[1], observed.shape[0])
    cf_pca = PCA(n_components=rank, svd_solver="randomized", random_state=seed)
    cf_obs = cf_pca.fit_transform(observed - cf_mean).astype(np.float32)
    cf_scale = float(cf_obs.std()) + 1e-9
    cf_obs /= cf_scale
    kept = float(cf_pca.explained_variance_ratio_.sum())
    console.print(
        f"[dim]collaborative factors {observed.shape[1]} -> {rank} components "
        f"({kept:.1%} of their variance)[/dim]"
    )

    cf_dim = rank
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
        pca_mean=pca.mean_.astype(np.float32),
        ridge_coef=np.asarray(ridge.coef_, dtype=np.float32),
        ridge_intercept=np.asarray(ridge.intercept_, dtype=np.float32),
    )


def save(art: FusionArtifacts) -> None:
    PATHS.ensure()
    np.savez_compressed(
        fused_path(),
        item_ids=art.item_ids,
        space=art.space,
        components=art.components,
        block_sizes=np.array(art.block_sizes),
        cf_r2=np.array([art.cf_r2]),
        cf_coverage=np.array([art.cf_coverage]),
        pca_mean=art.pca_mean if art.pca_mean is not None else np.zeros(0, dtype=np.float32),
        ridge_coef=art.ridge_coef if art.ridge_coef is not None else np.zeros(0, dtype=np.float32),
        ridge_intercept=(
            art.ridge_intercept if art.ridge_intercept is not None else np.zeros(0, dtype=np.float32)
        ),
    )


def load() -> FusionArtifacts:
    z = np.load(fused_path())

    def optional(key: str):
        return z[key] if key in z.files and z[key].size else None

    return FusionArtifacts(
        item_ids=z["item_ids"],
        space=z["space"],
        components=z["components"],
        block_sizes=tuple(int(x) for x in z["block_sizes"]),
        cf_r2=float(z["cf_r2"][0]),
        cf_coverage=float(z["cf_coverage"][0]),
        pca_mean=optional("pca_mean"),
        ridge_coef=optional("ridge_coef"),
        ridge_intercept=optional("ridge_intercept"),
    )


def exists() -> bool:
    return fused_path().exists()
