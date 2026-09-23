"""Per-app state, injected rather than captured.

The routes used to live inside create_app and close over ``engine`` and
``use_live`` lexically, which is why they could not move out of it. Module
state is not an option either: the tests construct several apps in one
process with different tokens and live settings, so a module-level engine
would be shared between them.

FastAPI dependencies solve both. Routers capture nothing at import time, so
they survive a module reload, and each app carries its own context.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request

from .. import store
from ..engine import Engine

# Below this many titles, assume the catalogue was never built rather than
# built small.
EMPTY_CATALOGUE = 25


def _catalogue_size() -> int:
    try:
        with store.session(read_only=True) as con:
            return int(con.execute("SELECT count(*) FROM titles").fetchone()[0])
    except Exception:
        return 0


@dataclass
class AppContext:
    """Everything a route needs that outlives a single request.

    ``engine`` is long-lived on purpose: it caches the feature space and the
    catalogue metadata, which together cost about a second to load. A fresh
    Engine per request would pay that every time.
    """

    engine: Engine
    use_live: bool
    token: str | None = None


def get_context(request: Request) -> AppContext:
    return request.app.state.ctx
