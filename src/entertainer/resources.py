"""How much of this machine the build may use.

The same build runs on an 8 GB laptop and on a server with several times
that, so memory is budgeted as a share of physical RAM rather than as fixed
numbers. The budget is enforced where a library accepts a limit (DuckDB) and
consulted where the build has a choice to make (whether to overlap stages).
Polars and NumPy take no limit, so for them it is a planning figure, not a
ceiling.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

#: Share of physical RAM the build plans to use when nothing overrides it.
#: Half leaves the machine usable for everything else while a build runs.
DEFAULT_MEMORY_FRACTION = 0.5
MEMORY_FRACTION_ENV = "ENTERTAINER_MEMORY_FRACTION"

GiB = 1024**3

#: Planning estimates of peak resident memory, not measurements. The
#: factorisation holds MovieLens-32M as a frame and a CSR matrix with a few
#: intermediate copies; the encoder holds a 0.6B-parameter model in fp32 plus
#: activations when it runs on the CPU or on unified memory (mps).
CF_PEAK_BYTES = 3 * GiB
ENCODER_HOST_PEAK_BYTES = {"cuda": 1 * GiB, "mps": 4 * GiB, "cpu": 4 * GiB}


def total_memory() -> int:
    """Physical RAM in bytes, or 0 when the platform will not say."""
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return 0


def memory_fraction(override: float | None = None) -> float:
    raw = override if override is not None else os.environ.get(MEMORY_FRACTION_ENV)
    if raw is None or raw == "":
        return DEFAULT_MEMORY_FRACTION
    fraction = float(raw)
    if not 0 < fraction <= 1:
        raise ValueError(f"memory fraction must be in (0, 1], got {fraction}")
    return fraction


@dataclass(frozen=True)
class Budget:
    total: int
    fraction: float

    @property
    def bytes(self) -> int:
        return int(self.total * self.fraction)

    def allows(self, need: int) -> bool:
        # An unknown total cannot rule anything out; behave as before budgets.
        return self.total == 0 or need <= self.bytes

    def describe(self) -> str:
        if not self.total:
            return "memory budget unknown"
        return (
            f"{self.bytes / GiB:.1f} GiB of {self.total / GiB:.0f} GiB RAM "
            f"({self.fraction:.0%})"
        )


def budget(fraction: float | None = None) -> Budget:
    return Budget(total=total_memory(), fraction=memory_fraction(fraction))


def duckdb_config() -> dict[str, str]:
    """DuckDB's own default is 80% of RAM, per process; the app and a build
    each open the database, so without a cap they can plan for 160%."""
    try:
        b = budget()
    except ValueError:
        # Every connection goes through here, the web app's included. A typo
        # in the variable is reported by `ent setup`, which validates it; it
        # must not take the whole app down.
        b = budget(DEFAULT_MEMORY_FRACTION)
    if not b.total:
        return {}
    return {"memory_limit": f"{max(b.bytes // (1024**2), 256)}MB"}


def can_overlap_cf(device: str, fraction: float | None = None) -> bool:
    """Whether the factorisation fits alongside the encoder in the budget.

    It starts after the catalogue build, whose Polars joins over IMDb are
    the build's largest allocation, so the overlap that matters is with the
    encoder.
    """
    need = CF_PEAK_BYTES + ENCODER_HOST_PEAK_BYTES.get(device, ENCODER_HOST_PEAK_BYTES["cpu"])
    return budget(fraction).allows(need)
