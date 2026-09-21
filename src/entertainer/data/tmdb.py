"""TMDB enrichment.

IMDb's bulk dumps give structure (who, when, how long, how well reviewed) but
no prose. TMDB supplies the plot synopsis, the curated keyword vocabulary and
a reliable ``original_language`` field — all three matter a great deal to a
text-embedding item tower, and the language field in particular is what makes
"recommend me a Malayalam thriller" reliable rather than approximate.

The ``/find`` endpoint is used because a single call resolves an IMDb id to a
full TMDB record, halving the request count versus find-then-detail.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import httpx
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from ..config import TMDB_API_BASE, tmdb_credentials

_console = Console()


def _tick(done: int, total: int, started: float, label: str) -> None:
    """Progress for a log file.

    Rich's progress bar renders nothing when stdout is not a terminal, and
    these passes run for hours in the background where a log file is the only
    way to see whether anything is happening.
    """
    if _console.is_terminal or done % 5000 or not done:
        return
    rate = done / max(time.monotonic() - started, 1e-6)
    remaining = (total - done) / rate if rate > 0 else 0
    print(
        f"{label}: {done:,}/{total:,} ({done / total:.1%}) "
        f"{rate:.0f}/s, ~{remaining / 60:.0f} min left",
        flush=True,
    )



# TMDB publishes no hard public rate limit any more but asks for restraint.
# 40 concurrent requests sits comfortably under the point where 429s appear.
DEFAULT_CONCURRENCY = 40
MAX_RETRIES = 5


def _auth() -> tuple[dict[str, str], dict[str, str]]:
    key, bearer = tmdb_credentials()
    headers = {"accept": "application/json"}
    params: dict[str, str] = {}
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    elif key:
        params["api_key"] = key
    else:
        raise RuntimeError("no TMDB credentials; set TMDB_BEARER or TMDB_API_KEY in .env")
    return headers, params


@dataclass
class EnrichResult:
    imdb_id: str
    tmdb_id: int | None = None
    kind: str | None = None
    overview: str | None = None
    tagline: str | None = None
    original_language: str | None = None
    original_title: str | None = None
    popularity: float | None = None
    tmdb_rating: float | None = None
    tmdb_votes: int | None = None
    poster_path: str | None = None
    genres: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    found: bool = False


# TMDB genre ids are a small fixed vocabulary; resolving them locally avoids a
# per-title round trip.
_GENRE_IDS = {
    28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy", 80: "Crime",
    99: "Documentary", 18: "Drama", 10751: "Family", 14: "Fantasy", 36: "History",
    27: "Horror", 10402: "Music", 9648: "Mystery", 10749: "Romance",
    878: "Science Fiction", 10770: "TV Movie", 53: "Thriller", 10752: "War",
    37: "Western", 10759: "Action & Adventure", 10762: "Kids", 10763: "News",
    10764: "Reality", 10765: "Sci-Fi & Fantasy", 10766: "Soap", 10767: "Talk",
    10768: "War & Politics",
}


class RateLimiter:
    """Simple token-bucket so bursts cannot trip TMDB's abuse protection."""

    def __init__(self, per_second: float):
        self._interval = 1.0 / per_second
        self._next = time.monotonic()
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            self._next = max(now, self._next) + self._interval
            delay = self._next - now
        if delay > 0:
            await asyncio.sleep(delay)


async def _get(
    client: httpx.AsyncClient, url: str, params: dict, limiter: RateLimiter
) -> dict | None:
    for attempt in range(MAX_RETRIES):
        await limiter.wait()
        try:
            r = await client.get(url, params=params, timeout=30)
        except (httpx.TimeoutException, httpx.TransportError):
            await asyncio.sleep(1.5 * (attempt + 1))
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code == 404:
            return None
        if r.status_code == 429:
            await asyncio.sleep(float(r.headers.get("retry-after", 2)) + 0.5)
            continue
        if 500 <= r.status_code < 600:
            await asyncio.sleep(1.5 * (attempt + 1))
            continue
        return None
    return None


def _parse_find(imdb_id: str, payload: dict | None) -> EnrichResult:
    out = EnrichResult(imdb_id=imdb_id)
    if not payload:
        return out
    for bucket, kind in (("movie_results", "movie"), ("tv_results", "tv")):
        hits = payload.get(bucket) or []
        if not hits:
            continue
        h = hits[0]
        out.found = True
        out.kind = kind
        out.tmdb_id = h.get("id")
        out.overview = (h.get("overview") or "").strip() or None
        out.original_language = h.get("original_language")
        out.original_title = h.get("original_title") or h.get("original_name")
        out.popularity = h.get("popularity")
        out.tmdb_rating = h.get("vote_average")
        out.tmdb_votes = h.get("vote_count")
        out.poster_path = h.get("poster_path")
        out.genres = [_GENRE_IDS[g] for g in (h.get("genre_ids") or []) if g in _GENRE_IDS]
        return out
    return out


async def _enrich_one(
    client: httpx.AsyncClient,
    imdb_id: str,
    params: dict,
    limiter: RateLimiter,
    want_keywords: bool,
) -> EnrichResult:
    payload = await _get(
        client,
        f"{TMDB_API_BASE}/find/{imdb_id}",
        {**params, "external_source": "imdb_id"},
        limiter,
    )
    res = _parse_find(imdb_id, payload)
    if want_keywords and res.found and res.tmdb_id:
        path = "movie" if res.kind == "movie" else "tv"
        kw = await _get(client, f"{TMDB_API_BASE}/{path}/{res.tmdb_id}/keywords", params, limiter)
        if kw:
            items = kw.get("keywords") if res.kind == "movie" else kw.get("results")
            res.keywords = [k["name"] for k in (items or []) if k.get("name")]
    return res


async def enrich_async(
    imdb_ids: Sequence[str],
    concurrency: int = DEFAULT_CONCURRENCY,
    keywords: bool = True,
    on_batch=None,
    batch_size: int = 500,
) -> list[EnrichResult]:
    concurrency = max(1, int(concurrency))
    headers, params = _auth()
    limiter = RateLimiter(per_second=concurrency)
    results: list[EnrichResult] = []
    pending: list[EnrichResult] = []

    limits = httpx.Limits(max_connections=concurrency + 10, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(headers=headers, limits=limits, http2=False) as client:

        with Progress(
            TextColumn("[bold blue]TMDB enrich"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as bar:
            task = bar.add_task("enrich", total=len(imdb_ids))
            started = time.monotonic()
            # Keep at most ``concurrency`` tasks alive. A semaphore alone only
            # limits active requests; creating 270k waiting Task objects first
            # can exhaust memory before the first response arrives.
            for start in range(0, len(imdb_ids), concurrency):
                batch = imdb_ids[start : start + concurrency]
                for res in await asyncio.gather(
                    *(_enrich_one(client, iid, params, limiter, keywords) for iid in batch)
                ):
                    results.append(res)
                    pending.append(res)
                    bar.advance(task)
                    _tick(len(results), len(imdb_ids), started, "enrich")
                    if on_batch and len(pending) >= batch_size:
                        on_batch(pending)
                        pending = []
            if on_batch and pending:
                on_batch(pending)
    return results


def enrich(imdb_ids: Iterable[str], **kw) -> list[EnrichResult]:
    return asyncio.run(enrich_async(list(imdb_ids), **kw))


def to_json_rows(results: Sequence[EnrichResult]) -> str:
    return "\n".join(json.dumps(r.__dict__) for r in results)


async def _keywords_only(
    client: httpx.AsyncClient, tmdb_id: int, kind: str, params: dict, limiter: RateLimiter
) -> tuple[int, list[str]]:
    path = "movie" if kind == "movie" else "tv"
    payload = await _get(client, f"{TMDB_API_BASE}/{path}/{tmdb_id}/keywords", params, limiter)
    if not payload:
        return tmdb_id, []
    items = payload.get("keywords") if kind == "movie" else payload.get("results")
    return tmdb_id, [k["name"] for k in (items or []) if k.get("name")]


async def backfill_keywords_async(
    targets: Sequence[tuple[int, int, str]],
    concurrency: int = DEFAULT_CONCURRENCY,
    on_batch=None,
    batch_size: int = 500,
) -> int:
    """Fetch keywords for already-enriched titles.

    Split out from the main pass because keywords double the request count and
    only earn their keep on titles a person might plausibly encounter. The
    long tail gets a synopsis and nothing else.

    ``targets``: (item_id, tmdb_id, kind) triples.
    """
    concurrency = max(1, int(concurrency))
    headers, params = _auth()
    limiter = RateLimiter(per_second=concurrency)
    pending: list[tuple[int, list[str]]] = []
    done = 0

    limits = httpx.Limits(max_connections=concurrency + 10, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(headers=headers, limits=limits) as client:

        with Progress(
            TextColumn("[bold blue]TMDB keywords"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as bar:
            task = bar.add_task("kw", total=len(targets))
            started = time.monotonic()
            for start in range(0, len(targets), concurrency):
                batch = targets[start : start + concurrency]
                resolved = await asyncio.gather(
                    *(_keywords_only(client, tmdb_id, kind, params, limiter) for _, tmdb_id, kind in batch)
                )
                for (item_id, _, _), (_tmdb_id, keywords) in zip(batch, resolved, strict=True):
                    pending.append((item_id, keywords))
                    done += 1
                    bar.advance(task)
                    _tick(done, len(targets), started, "keywords")
                    if on_batch and len(pending) >= batch_size:
                        on_batch(pending)
                        pending = []
            if on_batch and pending:
                on_batch(pending)
    return done


def backfill_keywords(targets, **kw) -> int:
    return asyncio.run(backfill_keywords_async(list(targets), **kw))


# --- direct lookup, for titles the catalogue never had ----------------------


async def _search_async(query: str, year: int | None, kind: str | None) -> list[dict]:
    headers, params = _auth()
    limiter = RateLimiter(per_second=10)
    out: list[dict] = []
    kinds = [kind] if kind in ("movie", "tv") else ["movie", "tv"]
    async with httpx.AsyncClient(headers=headers) as client:
        for k in kinds:
            p = {**params, "query": query, "include_adult": "false"}
            if year:
                p["year" if k == "movie" else "first_air_date_year"] = str(year)
            payload = await _get(client, f"{TMDB_API_BASE}/search/{k}", p, limiter)
            for hit in (payload or {}).get("results", [])[:8]:
                hit["_kind"] = k
                out.append(hit)
    out.sort(key=lambda h: -(h.get("popularity") or 0))
    return out


async def _detail_async(tmdb_id: int, kind: str) -> dict | None:
    headers, params = _auth()
    limiter = RateLimiter(per_second=10)
    path = "movie" if kind == "movie" else "tv"
    async with httpx.AsyncClient(headers=headers) as client:
        return await _get(
            client,
            f"{TMDB_API_BASE}/{path}/{tmdb_id}",
            {**params, "append_to_response": "keywords,credits,external_ids"},
            limiter,
        )


def search(query: str, year: int | None = None, kind: str | None = None) -> list[dict]:
    """Free-text TMDB search. Returns raw hits, most popular first."""
    return asyncio.run(_search_async(query, year, kind))


def detail(tmdb_id: int, kind: str) -> dict | None:
    return asyncio.run(_detail_async(tmdb_id, kind))


def detail_to_row(payload: dict, kind: str) -> dict:
    """Flatten a TMDB detail response into catalogue columns.

    Used by the add-a-missing-title path. Everything here is best-effort:
    a title obscure enough not to have made the vote floor is also likely to
    have partial metadata, and a half-filled row that can be rated is far
    more useful than a refusal.
    """
    ext = payload.get("external_ids") or {}
    credits = payload.get("credits") or {}
    crew = credits.get("crew") or []
    date = payload.get("release_date") or payload.get("first_air_date") or ""
    runtimes = payload.get("episode_run_time") or []

    return {
        "imdb_id": ext.get("imdb_id"),
        "tmdb_id": payload.get("id"),
        "kind": kind,
        "title": payload.get("title") or payload.get("name"),
        "original_title": payload.get("original_title") or payload.get("original_name"),
        "year": int(date[:4]) if date[:4].isdigit() else None,
        "runtime": payload.get("runtime") or (runtimes[0] if runtimes else None),
        "genres": [g["name"] for g in (payload.get("genres") or []) if g.get("name")],
        "language": payload.get("original_language"),
        "languages": [
            sl["iso_639_1"] for sl in (payload.get("spoken_languages") or []) if sl.get("iso_639_1")
        ],
        "countries": [
            c["iso_3166_1"] for c in (payload.get("production_countries") or []) if c.get("iso_3166_1")
        ],
        "tmdb_rating": payload.get("vote_average"),
        "tmdb_votes": payload.get("vote_count"),
        "popularity": payload.get("popularity"),
        "directors": [
            c["name"] for c in crew if c.get("job") in ("Director",) and c.get("name")
        ][:4],
        "writers": [
            c["name"] for c in crew if c.get("department") == "Writing" and c.get("name")
        ][:4],
        "cast_names": [c["name"] for c in (credits.get("cast") or [])[:8] if c.get("name")],
        "keywords": [
            k["name"]
            for k in ((payload.get("keywords") or {}).get("keywords")
                      or (payload.get("keywords") or {}).get("results") or [])
            if k.get("name")
        ],
        "overview": (payload.get("overview") or "").strip() or None,
        "tagline": (payload.get("tagline") or "").strip() or None,
        "poster_path": payload.get("poster_path"),
        "adult": bool(payload.get("adult")),
    }
