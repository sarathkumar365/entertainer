"""A TMDB-backed feed, for a machine with no catalogue.

The rating interface normally reads a locally built catalogue. That is the
better source — it is IMDb-backed, so it knows about 285 Malayalam films from
the last two years where TMDB's vote counts only support about five — but it
is a 21MB file that has to be moved between machines, and moving it is
friction at exactly the moment someone is willing to sit and rate things.

This mode removes the file. It asks TMDB's discover endpoint for recent,
well-received titles per language and serves those directly. Verdicts are
recorded against a catalogue row created on demand, so they are real events
in the normal event log and merge cleanly into the main machine's catalogue
later via `ent export` / `ent import`, which key on the IMDb id.

The trade is coverage, and it is a real one: TMDB's rating pool for South
Indian cinema is two orders of magnitude thinner than IMDb's, so the feed is
shallower and skews to whatever had international distribution. Use the
bundle when you can; use this when moving a file is the thing stopping you.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx

from ..config import TMDB_API_BASE, language_label, tmdb_credentials

# TMDB vote counts vary enormously by industry, so a single floor either
# floods the feed with Hollywood or returns nothing for Kannada. These are
# calibrated to return roughly comparable cultural standing.
VOTE_FLOORS: dict[str, int] = {
    "en": 800, "ja": 120, "ko": 120, "hi": 60, "es": 100, "fr": 100,
    "de": 60, "it": 60, "zh": 60, "cn": 40, "pt": 50, "ru": 50, "tr": 40,
    "th": 30, "id": 30, "ml": 25, "ta": 40, "te": 40, "kn": 15, "bn": 15,
    "mr": 15, "fa": 20, "sv": 30, "da": 30, "no": 25, "fi": 20, "pl": 30,
    "nl": 30,
}
DEFAULT_FLOOR = 40


def _auth() -> tuple[dict[str, str], dict[str, str]]:
    key, bearer = tmdb_credentials()
    headers = {"accept": "application/json"}
    params: dict[str, str] = {}
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    elif key:
        params["api_key"] = key
    else:
        raise RuntimeError("no TMDB credentials")
    return headers, params


@dataclass
class LiveRequest:
    languages: dict[str, float]
    since: str
    limit: int = 60
    page: int = 1
    kind: str = "movie"


async def _discover(
    client: httpx.AsyncClient, lang: str, req: LiveRequest, params: dict, want: int
) -> list[dict]:
    path = "movie" if req.kind != "tv" else "tv"
    date_field = "primary_release_date.gte" if path == "movie" else "first_air_date.gte"
    out: list[dict] = []
    # Ask for as many pages as the quota needs; TMDB returns 20 per page.
    for page in range(req.page, req.page + max(1, (want // 20) + 1)):
        try:
            r = await client.get(
                f"{TMDB_API_BASE}/discover/{path}",
                params={
                    **params,
                    "with_original_language": lang,
                    date_field: req.since,
                    "vote_count.gte": VOTE_FLOORS.get(lang, DEFAULT_FLOOR),
                    "sort_by": "vote_average.desc",
                    "include_adult": "false",
                    "page": page,
                },
                timeout=20,
            )
            if r.status_code != 200:
                break
            results = r.json().get("results", [])
        except httpx.HTTPError:
            break
        if not results:
            break
        for m in results:
            if not m.get("poster_path"):
                continue
            date = m.get("release_date") or m.get("first_air_date") or ""
            out.append(
                {
                    "tmdb_id": m.get("id"),
                    "kind": path,
                    "title": m.get("title") or m.get("name"),
                    "original_title": m.get("original_title") or m.get("original_name"),
                    "year": int(date[:4]) if date[:4].isdigit() else None,
                    "language": lang,
                    "language_name": language_label(lang),
                    "overview": (m.get("overview") or "")[:260] or None,
                    "poster_path": m.get("poster_path"),
                    "rating": m.get("vote_average"),
                    "votes": m.get("vote_count"),
                }
            )
        if len(out) >= want:
            break
    return out[:want]


async def _fetch_async(req: LiveRequest) -> list[dict]:
    headers, params = _auth()
    total = sum(w for w in req.languages.values() if w > 0) or 1.0
    quota = {
        lang: max(1, round((w / total) * req.limit))
        for lang, w in req.languages.items()
        if w > 0
    }
    async with httpx.AsyncClient(headers=headers) as client:
        results = await asyncio.gather(
            *(_discover(client, lang, req, params, want) for lang, want in quota.items()),
            return_exceptions=True,
        )

    buckets = {
        lang: (r if isinstance(r, list) else [])
        for lang, r in zip(quota, results, strict=True)
    }
    # Interleave, same reasoning as the catalogue feed: a screen that arrives
    # in language blocks loses its balance the moment someone stops halfway.
    out: list[dict] = []
    order = sorted(buckets, key=lambda lang: -len(buckets[lang]))
    while any(buckets.values()) and len(out) < req.limit:
        for lang in order:
            if buckets[lang]:
                out.append(buckets[lang].pop(0))
    return out[: req.limit]


def fetch(req: LiveRequest) -> list[dict]:
    return asyncio.run(_fetch_async(req))
