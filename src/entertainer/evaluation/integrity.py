"""The holdout contract.

An offline benchmark is only honest if the users being replayed were never
seen by the model replaying them. The population prior is fitted on
MovieLens; the benchmark replays MovieLens users; so a prior fitted without
withholding anyone has already read the answers.

Both halves of that contract lived in cli.py, about 1100 lines apart: `ent
data cf` chose the held-out users and wrote them to disk, and `ent eval`
checked the file was there before reporting a number. Neither referred to the
other, and the filename was written out three times. Keeping them in one
module is the point — a contract split across a file is one edit away from
being silently broken.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import PATHS
from ..errors import IntegrityRefusal

FILENAME = "cf_holdout_users.npy"


def holdout_path() -> Path:
    return PATHS.artifacts / FILENAME


def choose_holdout(users: np.ndarray, size: int, seed: int = 0) -> np.ndarray | None:
    """Pick the users to withhold from collaborative-filtering training.

    Seeded, so a rebuild withholds the same users and the benchmark stays
    comparable across runs. ``size`` of zero means withhold nobody, which is
    legitimate for a local build that will never be benchmarked — but see
    ``require_holdout``, which then refuses to report a number.
    """
    if not size:
        return None
    rng = np.random.default_rng(seed)
    return rng.choice(users, size=min(size, len(users)), replace=False)


def save_holdout(users: np.ndarray) -> Path:
    path = holdout_path()
    PATHS.ensure()
    np.save(path, users.astype(np.int32))
    return path


def forget_holdout() -> None:
    """Drop the record before a refit.

    A refit that withholds nobody writes no new record, and the old one would
    otherwise go on vouching for factors that were trained on those users.
    """
    holdout_path().unlink(missing_ok=True)


def load_holdout() -> np.ndarray | None:
    path = holdout_path()
    return np.load(path) if path.exists() else None


def holdout_recorded() -> bool:
    return holdout_path().exists()


def benchmark_users() -> np.ndarray:
    """The only MovieLens users a benchmark may replay: the ones held out.

    Everyone else trained the CF factors, and through them the fused space
    every arm ranks in, and the prior. Checking that the record exists is not
    enough on its own — the replayed users must be drawn from it.
    """
    require_holdout()
    return load_holdout()


def require_holdout() -> None:
    """Refuse to benchmark a prior that may have seen the replayed users.

    Raises rather than warning. A warning next to a plausible-looking NDCG
    gets read as a caveat on a real number; there is no real number here.
    """
    if not holdout_recorded():
        raise IntegrityRefusal(
            "no record of which users were held out; the prior may contain the very "
            "users being replayed. Refusing to report a number."
        )
