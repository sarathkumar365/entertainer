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
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import store
from ..config import has_tmdb, language_label
from ..engine import Engine
from ..models.taste import VERDICTS
from ..resolve import search as resolve_search
from .feed import DEFAULT_WEIGHTS, FeedRequest, fetch

STATIC = Path(__file__).parent / "static"
POSTER_BASE = "https://image.tmdb.org/t/p/w342"

# Below this many titles, assume the catalogue was never built rather than
# built small.
EMPTY_CATALOGUE = 25


def _poster(path: str | None) -> str | None:
    return f"{POSTER_BASE}{path}" if path else None


def _present(row: dict) -> dict:
    """Shape a catalogue row for the interface."""
    return {
        "item_id": row.get("item_id"),
        "title": row.get("title"),
        "original_title": (
            row.get("original_title")
            if row.get("original_title") and row.get("original_title") != row.get("title")
            else None
        ),
        "year": row.get("year"),
        "kind": row.get("kind"),
        "language": row.get("language"),
        "language_name": language_label(row.get("language")),
        "runtime": row.get("runtime"),
        "genres": list(row.get("genres") or [])[:3],
        "directors": list(row.get("directors") or [])[:2],
        "rating": row.get("imdb_rating"),
        "votes": row.get("imdb_votes"),
        "overview": (row.get("overview") or "")[:260] or None,
        "poster": _poster(row.get("poster_path")),
    }


class Verdict(BaseModel):
    item_id: int
    verdict: str = Field(description="love | like | ok | meh | dislike | hate | unseen")


class AddRequest(BaseModel):
    tmdb_id: int
    kind: str = "movie"
    verdict: str | None = None


class ValidationSealRequest(BaseModel):
    item_ids: list[int] = Field(min_length=20, max_length=100)


class ValidationRevealRequest(BaseModel):
    verdict: str = Field(description="love | like | ok | meh | dislike | hate | unseen")


def _catalogue_size() -> int:
    try:
        with store.session(read_only=True) as con:
            return int(con.execute("SELECT count(*) FROM titles").fetchone()[0])
    except Exception:
        return 0


def create_app(token: str | None = None, live: bool | None = None) -> FastAPI:
    """Build the app. A token is required once it is bound off localhost.

    The interface writes to the verdict log and can pull titles from TMDB, so
    on anything but the loopback interface it needs a shared secret. This is
    a single-user tool on a home network, not a service — a token in the URL
    is the right weight of protection, and the alternative people actually
    reach for is no protection at all.
    """
    app = FastAPI(title="entertainer", docs_url=None, redoc_url=None)
    engine = Engine()
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
    use_live = live if live is not None else _catalogue_size() < EMPTY_CATALOGUE

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

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/languages")
    def languages() -> dict[str, Any]:
        if use_live:
            from .live import VOTE_FLOORS

            codes = [c for c in DEFAULT_WEIGHTS if c in VOTE_FLOORS]
            return {
                "languages": [
                    {
                        "code": c,
                        "name": language_label(c),
                        "titles": 0,
                        "weight": DEFAULT_WEIGHTS.get(c, 0.0),
                    }
                    for c in codes
                ]
            }
        with store.session(read_only=True) as con:
            rows = con.execute(
                """
                SELECT language, count(*) n FROM titles
                WHERE poster_path IS NOT NULL AND year >= ?
                GROUP BY 1 HAVING n >= 20 ORDER BY n DESC, language ASC
                """,
                [dt.date.today().year - 6],
            ).fetchall()
        return {
            "languages": [
                {
                    "code": code,
                    "name": language_label(code),
                    "titles": int(n),
                    "weight": DEFAULT_WEIGHTS.get(code, 0.0),
                }
                for code, n in rows
                if code != "xx"
            ]
        }

    @app.get("/api/mode")
    def mode() -> dict[str, Any]:
        return {
            "live": use_live,
            "catalogue": _catalogue_size(),
            "tmdb": has_tmdb(),
        }

    @app.get("/api/feed")
    def feed(
        years: int = 2,
        limit: int = 60,
        langs: str = "",
        kind: str = "",
        min_quality: float = 0.0,
        page: int = 0,
    ) -> dict[str, Any]:
        weights = dict(DEFAULT_WEIGHTS)
        if langs.strip():
            weights = {}
            for part in langs.split(","):
                if not part.strip():
                    continue
                code, _, w = part.partition(":")
                try:
                    weights[code.strip()] = float(w) if w else 1.0
                except ValueError:
                    weights[code.strip()] = 1.0
        if use_live:
            if not has_tmdb():
                raise HTTPException(
                    400, "no catalogue and no TMDB credentials — nothing to show"
                )
            from .live import LiveRequest
            from .live import fetch as live_fetch

            since = f"{dt.date.today().year - max(years, 0)}-01-01"
            items = live_fetch(
                LiveRequest(
                    languages=weights,
                    since=since,
                    # Over-fetch because filtering locally saved ratings below
                    # can remove part of TMDB's first result page.
                    limit=max(1, min(limit * 3, 200)),
                    page=page + 1,
                    kind=kind or "movie",
                )
            )
            with store.session(read_only=True) as con:
                rated_tmdb = {
                    int(row[0])
                    for row in con.execute(
                        """
                        SELECT DISTINCT t.tmdb_id FROM events e JOIN titles t USING (item_id)
                        WHERE e.kind = 'rate' AND t.tmdb_id IS NOT NULL
                        """
                    ).fetchall()
                }
            items = [it for it in items if int(it.get("tmdb_id") or -1) not in rated_tmdb]
            for it in items:
                it["poster"] = _poster(it.pop("poster_path", None))
                it["external"] = True
            return {"items": items[:limit], "live": True}

        req = FeedRequest(
            years=years,
            limit=max(1, min(limit, 200)),
            languages=weights,
            min_quality=min_quality,
            kind=kind or None,
            offset_seed=page,
        )
        with store.session(read_only=True) as con:
            rows = fetch(con, req, dt.date.today().year)
        return {"items": [_present(r) for r in rows], "live": False}

    @app.post("/api/rate")
    def rate(body: Verdict) -> dict[str, Any]:
        with store.session(read_only=True) as con:
            sealed = con.execute(
                "SELECT 1 FROM validation_cases WHERE item_id = ? AND status = 'sealed'",
                [body.item_id],
            ).fetchone()
        if sealed:
            raise HTTPException(400, "this title is a sealed validation case; reveal it through validation")
        if body.verdict == "unseen":
            with store.session() as con:
                store.log_event(con, body.item_id, "unseen", None, "web", {"answer": "unseen"})
            return {"ok": True, "verdict": "unseen"}
        if body.verdict not in VERDICTS:
            raise HTTPException(400, f"unknown verdict {body.verdict!r}")
        with store.session() as con:
            engine.record(con, body.item_id, body.verdict, source="web")
        return {"ok": True, "verdict": body.verdict}

    @app.post("/api/undo")
    def undo() -> dict[str, Any]:
        with store.session() as con:
            row = con.execute(
                "SELECT event_id, item_id FROM events WHERE source = 'web' "
                "ORDER BY ts DESC LIMIT 1"
            ).fetchone()
            if not row:
                return {"ok": False}
            con.execute("DELETE FROM events WHERE event_id = ?", [row[0]])
            title = con.execute(
                "SELECT title FROM titles WHERE item_id = ?", [row[1]]
            ).fetchone()
        return {"ok": True, "item_id": int(row[1]), "title": title[0] if title else None}

    @app.get("/api/progress")
    def progress() -> dict[str, Any]:
        with store.session(read_only=True) as con:
            counts = store.counts(con)
            by_lang = con.execute(
                """
                SELECT t.language, count(DISTINCT e.item_id) n
                FROM events e JOIN titles t USING (item_id)
                WHERE e.kind = 'rate' GROUP BY 1 ORDER BY n DESC, t.language ASC LIMIT 12
                """
            ).fetchall()
            recent = con.execute(
                """
                SELECT t.title, t.year, e.context FROM events e JOIN titles t USING (item_id)
                WHERE e.kind = 'rate' ORDER BY e.ts DESC LIMIT 8
                """
            ).fetchall()
        import json as _json

        return {
            "rated": counts["ratings"],
            "events": counts["events"],
            "titles": counts["titles"],
            "by_language": [
                {"code": c, "name": language_label(c), "count": int(n)} for c, n in by_lang
            ],
            "recent": [
                {
                    "title": t,
                    "year": y,
                    "verdict": (_json.loads(ctx or "{}") or {}).get("verdict"),
                }
                for t, y, ctx in recent
            ],
        }

    @app.get("/api/rated")
    def rated_titles(limit: int = 200) -> dict[str, Any]:
        """Latest explicit verdict per title, most recently changed first."""
        with store.session(read_only=True) as con:
            cur = con.execute(
                """
                SELECT t.*, e.value, e.context, e.ts FROM titles t JOIN (
                    SELECT item_id, value, context, ts,
                           row_number() OVER (PARTITION BY item_id ORDER BY ts DESC) rn
                    FROM events WHERE kind = 'rate'
                ) e USING (item_id)
                WHERE e.rn = 1 ORDER BY e.ts DESC LIMIT ?
                """,
                [max(1, min(limit, 500))],
            )
            columns = [d[0] for d in cur.description]
            rows = [dict(zip(columns, row, strict=True)) for row in cur.fetchall()]
        import json as _json

        return {
            "items": [
                {
                    **_present(row),
                    "verdict": (_json.loads(row.get("context") or "{}") or {}).get("verdict"),
                    "rated_at": str(row.get("ts") or ""),
                }
                for row in rows
            ]
        }

    @app.post("/api/validation/seal")
    def seal_validation(body: ValidationSealRequest) -> dict[str, Any]:
        """Seal a user-selected watched pool before any verdict is revealed."""
        from ..evaluation import personal

        try:
            return personal.seal(engine, body.item_ids)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/validation/{case_id}/reveal")
    def reveal_validation(case_id: str, body: ValidationRevealRequest) -> dict[str, Any]:
        """Reveal exactly one sealed validation verdict and add it to training."""
        from ..evaluation import personal

        try:
            return personal.reveal(engine, case_id, body.verdict)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/validation/summary")
    def validation_summary() -> dict[str, Any]:
        """Interval-aware personal evidence; no bare point-estimate claims."""
        from ..evaluation import personal

        return personal.summary()

    @app.get("/api/predict/{item_id}")
    def predict(item_id: int) -> dict[str, Any]:
        """Score a catalogue title without recording or changing a verdict."""
        from ..evaluation import personal

        try:
            return personal.predict(engine, item_id)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/recommendations/slate")
    def recommendation_slate(k: int = 10) -> dict[str, Any]:
        """Create and log an observational production slate with propensities."""
        from ..recommend import Filters, recommend

        with store.session() as con:
            # Recommendations are derived from the event log on every call;
            # do not persist a transient model merely to create a slate.
            model = engine.fit(con, save=False)
            if model is None:
                raise HTTPException(409, "at least three explicit verdicts are needed before recommendations")
            fs = engine.features(con)
            meta = engine.meta(con)
            slate_id = store.new_slate_id()
            recs = recommend(
                model, fs, meta, k=max(1, min(k, 20)), strategy="thompson",
                filters=Filters(exclude=frozenset(store.interacted(con))),
            )
            store.log_impressions(
                con, slate_id,
                [(r.item_id, r.position, r.score, r.propensity, r.explored) for r in recs],
                policy="bayesian-thompson-v1",
            )
        return {
            "slate_id": slate_id,
            "observational": True,
            "items": [{**_present(meta[r.item_id]), "score": r.mean * 10.0, "std": r.std * 10.0,
                       "propensity": r.propensity, "explored": r.explored} for r in recs],
        }

    @app.get("/api/search")
    def search(q: str, limit: int = 12) -> dict[str, Any]:
        """Catalogue first, then TMDB for anything the catalogue lacks."""
        if not q.strip():
            return {"catalogue": [], "tmdb": []}
        with store.session(read_only=True) as con:
            hits = resolve_search(con, q, limit=limit)
            rows = store.item_rows(con, [h.item_id for h in hits])
        catalogue = [_present(rows[h.item_id]) for h in hits if h.item_id in rows]

        external: list[dict] = []
        if has_tmdb():
            from ..data import tmdb

            known_tmdb = {c.get("tmdb_id") for c in catalogue}
            try:
                for hit in tmdb.search(q)[:limit]:
                    if hit.get("id") in known_tmdb:
                        continue
                    date = hit.get("release_date") or hit.get("first_air_date") or ""
                    external.append(
                        {
                            "tmdb_id": hit.get("id"),
                            "kind": hit["_kind"],
                            "title": hit.get("title") or hit.get("name"),
                            "original_title": hit.get("original_title")
                            or hit.get("original_name"),
                            "year": int(date[:4]) if date[:4].isdigit() else None,
                            "language": hit.get("original_language"),
                            "language_name": language_label(hit.get("original_language")),
                            "overview": (hit.get("overview") or "")[:260] or None,
                            "poster": _poster(hit.get("poster_path")),
                            "rating": hit.get("vote_average"),
                            "votes": hit.get("vote_count"),
                        }
                    )
            except Exception as exc:  # pragma: no cover - network dependent
                external = []
                return {"catalogue": catalogue, "tmdb": external, "tmdb_error": str(exc)}
        return {"catalogue": catalogue, "tmdb": external}

    @app.post("/api/add")
    def add(body: AddRequest) -> dict[str, Any]:
        """Pull a title TMDB knows into the catalogue and place it in the item space."""
        import numpy as np

        from ..data import catalog, tmdb
        from ..models import encoder, fusion
        from ..models.itemcard import build_card

        if not has_tmdb():
            raise HTTPException(400, "no TMDB credentials configured")
        payload = tmdb.detail(body.tmdb_id, body.kind)
        if not payload:
            raise HTTPException(404, "TMDB returned nothing for that id")
        row = tmdb.detail_to_row(payload, body.kind)

        art = None
        if not use_live:
            art = fusion.load()
            if art.pca_mean is None:
                raise HTTPException(
                    500, "item space predates projection; run `ent data fuse`"
                )

        with store.session() as con:
            item_id = catalog.insert_title(con, row)
            already = con.execute(
                "SELECT count(*) FROM titles WHERE item_id = ?", [item_id]
            ).fetchone()[0]
        del already

        # Placing the title in the item space needs the encoder, which is a
        # 1.2GB download. On a machine that only collects verdicts that is a
        # steep price for something the main machine will redo anyway, so it
        # is skipped when the space is not present. The verdict is still
        # recorded and still merges back, because it keys on the IMDb id.
        if not use_live and item_id not in engine.features().index:
            content = encoder.encode_texts([build_card(row)], show_progress=False)
            try:
                latent = art.project(content)
            except RuntimeError as exc:
                raise HTTPException(500, str(exc)) from exc
            ids, mat = encoder.load()
            encoder.save(np.append(ids, np.int32(item_id)), np.vstack([mat, content]))
            art.item_ids = np.append(art.item_ids, np.int32(item_id))
            art.space = np.vstack([art.space, latent])
            fusion.save(art)
            engine._fs = None
            engine._meta = None

        if body.verdict:
            with store.session() as con:
                if body.verdict == "unseen":
                    store.log_event(con, item_id, "unseen", None, "web")
                else:
                    engine.record(con, item_id, body.verdict, source="web")

        with store.session(read_only=True) as con:
            rows = store.item_rows(con, [item_id])
        return {"ok": True, "item": _present(rows[item_id])}

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
