"""Command groups.

A Typer command exists only once its module has been imported and its
decorator has run, so these imports are load-bearing rather than decorative —
that is what the noqa records.

The order here is the order `ent --help` lists the commands in.
"""

from __future__ import annotations

from . import (
    browse,  # noqa: F401
    build,  # noqa: F401
    diagnose,  # noqa: F401
    release,  # noqa: F401
    serve,  # noqa: F401
    transfer,  # noqa: F401
    verdicts,  # noqa: F401
)
from ._apps import app, bundle_app, data_app, release_app

__all__ = ["app", "bundle_app", "data_app", "release_app"]
