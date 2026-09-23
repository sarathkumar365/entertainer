"""A local rating interface.

The engine learns from verdicts and nothing else, so the rate-limiting step
in making it good is how quickly a person can give verdicts. A terminal is a
poor instrument for that: recalling and typing a transliterated title is
slower and more error-prone than recognising a poster, and the friction
compounds over the hundred-odd ratings the model actually needs.

This serves a grid of posters on localhost. Nothing leaves the machine except
TMDB requests for catalogue metadata, exactly as the CLI already makes.

Everything written here goes into the same event log the CLI uses, so
`ent recs`, `ent taste` and `ent audit` see these verdicts immediately.

The routes live in ``routers``; this module only assembles them.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import store
from ..engine import Engine
from .context import EMPTY_CATALOGUE, AppContext, _catalogue_size
from .present import POSTER_BASE
from .routers import ALL as ROUTERS

STATIC = Path(__file__).parent / "static"

# Re-exported: cli.py imports EMPTY_CATALOGUE and _catalogue_size from here to
# make the same live-versus-catalogue decision before it binds a port.
__all__ = ["EMPTY_CATALOGUE", "POSTER_BASE", "STATIC", "_catalogue_size", "create_app"]


def create_app(token: str | None = None, live: bool | None = None) -> FastAPI:
    """Build the app. A token is required once it is bound off localhost.

    The interface writes to the verdict log and can pull titles from TMDB, so
    on anything but the loopback interface it needs a shared secret. This is
    a single-user tool on a home network, not a service — a token in the URL
    is the right weight of protection, and the alternative people actually
    reach for is no protection at all.

    State hangs off ``app.state`` rather than being closed over, because the
    tests construct several apps in one process with different tokens and
    live settings. A module-level engine would be shared between them.
    """
    app = FastAPI(title="entertainer", docs_url=None, redoc_url=None)

    # Apply idempotent schema additions before any read-only endpoint is hit.
    # Existing profiles therefore gain the evidence ledger without a manual
    # migration command and without rewriting their event history.
    with store.session():
        pass

    # Live mode when asked for, and automatically when there is essentially no
    # catalogue to read — a fresh clone should be able to rate straight away
    # rather than showing an empty grid and an instruction to fetch a file.
    # The threshold is "empty", not "small": a deliberately narrow catalogue
    # is still a catalogue and must not be silently overridden.
    app.state.ctx = AppContext(
        engine=Engine(),
        use_live=live if live is not None else _catalogue_size() < EMPTY_CATALOGUE,
        token=token,
    )

    if token:

        @app.middleware("http")
        async def require_token(request: Request, call_next):
            supplied = (
                request.query_params.get("token")
                or request.headers.get("x-entertainer-token")
                or request.cookies.get("entertainer_token")
            )
            if supplied != token:
                return JSONResponse({"detail": "bad or missing token"}, status_code=401)
            response = await call_next(request)
            if request.query_params.get("token") == token:
                # Set once from the initial link so in-page fetches carry it.
                response.set_cookie(
                    "entertainer_token", token, httponly=True, samesite="lax"
                )
            return response

    for module in ROUTERS:
        app.include_router(module.router)

    # Mounted last, so the token middleware above covers /static too.
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
