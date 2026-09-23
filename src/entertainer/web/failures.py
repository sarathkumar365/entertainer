"""Turning expected failures into answers the interface can render.

Two states are normal on a machine that has not finished a build: the model
artefacts do not exist yet, and a running build owns the database lock.
Neither is a defect, but both used to surface as a bare 500 with the body
``Internal Server Error`` — indistinguishable from a crash, and with nothing
in it a page could act on.

So the domain errors get status codes that say what kind of state this is,
and a ``code`` the interface can branch on without matching English prose.
Anything genuinely unexpected still answers as JSON rather than plain text,
because a page that cannot parse the error shows nothing at all.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ..errors import (
    CatalogueBusy,
    EntertainerError,
    MissingCatalogue,
    ModelNotReady,
    NotEnoughEvidence,
)

log = logging.getLogger("entertainer.web")

#: How long a client should wait before retrying a 503. A build runs for
#: hours, so this paces polling rather than promising a deadline.
RETRY_AFTER = "30"

#: Domain error -> (status, code, what the browser is told). Order matters:
#: the first class that matches wins, so subclasses precede their bases.
#:
#: The message is rewritten rather than passed through because the one the
#: error carries is written for a terminal and names the command that fixes
#: it. This page is the alternative to a terminal; telling someone who opened
#: a browser to go and type something is the opposite of what it is for.
_MAPPING: tuple[tuple[type[EntertainerError], int, str, str], ...] = (
    (
        ModelNotReady, 503, "model_not_ready",
        "The catalogue is here, but the model that scores it is still "
        "unfinished. This page works as soon as the build completes.",
    ),
    (
        CatalogueBusy, 503, "catalogue_busy",
        "A build is using the library right now. This page works again as "
        "soon as it finishes.",
    ),
    (
        MissingCatalogue, 503, "no_catalogue",
        "There is no catalogue on this machine yet.",
    ),
    (
        NotEnoughEvidence, 409, "not_enough_evidence",
        "There is not enough here yet to answer that. Rate a few more titles.",
    ),
)


def classify(exc: EntertainerError) -> tuple[int, str, str]:
    """The status, machine-readable code and message for one domain error.

    An error with no entry of its own keeps its own words: those are the
    refusals written for whoever asked, and there is nothing generic to say
    in their place.
    """
    for kind, status, code, message in _MAPPING:
        if isinstance(exc, kind):
            return status, code, message
    return 409, "refused", str(exc)


def install(app: FastAPI) -> None:
    """Register the handlers. Safe to call on any app in this package."""

    @app.exception_handler(EntertainerError)
    async def _expected(request: Request, exc: EntertainerError) -> JSONResponse:
        status, code, message = classify(exc)
        headers = {"Retry-After": RETRY_AFTER} if status == 503 else None
        return JSONResponse(
            {"detail": message, "code": code}, status_code=status, headers=headers
        )

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        # Logged rather than returned: the message may name a path or a
        # query, and the page has nothing useful to do with a traceback.
        log.exception("unhandled error serving %s %s", request.method, request.url.path)
        return JSONResponse(
            {"detail": "Something went wrong on this machine.", "code": "internal"},
            status_code=500,
        )
