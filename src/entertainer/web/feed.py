"""Choosing which titles to put in front of someone for rating.

The feed is not a recommendation list and must not be built like one. Its job
is to collect verdicts efficiently, which means showing titles the person is
likely to *have an opinion about* — recent, well regarded within their own
industry, and spread across languages so the resulting profile is not an
accident of whichever cinema happens to dominate the catalogue.

"Well regarded within their own industry" is doing real work here. An
absolute rating cut would return almost nothing in Kannada and a wall of
English, because the vote populations differ by two orders of magnitude. The
`quality` column is already standardised per language (see
``catalog.recalibrate_quality``), so the same threshold means the same thing
everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# How much of the feed each language gets, relative to the others. These are
# starting points the user overrides in the interface; they exist so the first
# screen is already roughly right rather than uniformly wrong.
DEFAULT_WEIGHTS: dict[str, float] = {
    "ml": 1.0,
    "ta": 1.0,
    "en": 1.0,
    "ko": 0.8,
    "ja": 0.7,
    "hi": 0.7,
    "kn": 0.6,
    "te": 0.25,   # the user watches little Telugu; present but not prominent
    "es": 0.3,
    "fr": 0.3,
    "zh": 0.25,
    "th": 0.2,
    "it": 0.2,
    "de": 0.2,
    "bn": 0.2,
    "tr": 0.15,
    "id": 0.15,
    "pt": 0.15,
    "fa": 0.15,
    "sv": 0.1,
    "da": 0.1,
    "pl": 0.1,
    "ru": 0.1,
    "cn": 0.2,
}


@dataclass
class FeedRequest:
    years: int = 2
    limit: int = 60
    languages: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    min_quality: float = 0.0
    kind: str | None = None
    exclude_seen: bool = True
    offset_seed: int = 0


SELECT = """
    SELECT item_id, title, original_title, year, kind, language, runtime,
           genres, imdb_rating, imdb_votes, quality, overview, poster_path, directors
"""


def _allocate(weights: dict[str, float], limit: int) -> dict[str, int]:
    """Largest-remainder allocation, so small weights still get whole slots."""
    total = sum(w for w in weights.values() if w > 0)
    if total <= 0:
        return {}
    exact = {lang: (w / total) * limit for lang, w in weights.items() if w > 0}
    base = {lang: int(v) for lang, v in exact.items()}
    remainder = limit - sum(base.values())
    for lang, _ in sorted(exact.items(), key=lambda kv: -(kv[1] - int(kv[1]))):
        if remainder <= 0:
            break
        base[lang] += 1
        remainder -= 1
    return {lang: n for lang, n in base.items() if n > 0}


def fetch(con, req: FeedRequest, current_year: int) -> list[dict]:
    """Return an interleaved, language-balanced set of titles to rate."""
    cutoff = current_year - max(req.years, 0)
    quota = _allocate(req.languages, req.limit)
    if not quota:
        return []

    seen_clause = (
        " AND item_id NOT IN (SELECT DISTINCT item_id FROM events)"
        if req.exclude_seen
        else ""
    )
    kind_clause = " AND kind = ?" if req.kind in ("movie", "tv") else ""

    by_lang: dict[str, list[dict]] = {}
    for lang, want in quota.items():
        params: list = [lang, cutoff, req.min_quality]
        if kind_clause:
            params.append(req.kind)
        # Over-fetch, then take a window, so repeated visits are not identical.
        params.append(want * 3)
        cur = con.execute(
            f"""{SELECT}
            FROM titles
            WHERE language = ?
              AND year >= ?
              AND quality >= ?
              AND poster_path IS NOT NULL
              {kind_clause}{seen_clause}
            ORDER BY quality DESC
            LIMIT ?
            """,
            params,
        )
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]
        start = (req.offset_seed * want) % max(len(rows), 1)
        rotated = rows[start:] + rows[:start]
        by_lang[lang] = rotated[:want]

    # Interleave so the grid alternates languages rather than arriving in
    # blocks — the point of the balance is lost if someone stops halfway down
    # a screen of one industry.
    out: list[dict] = []
    order = sorted(by_lang, key=lambda lang: -len(by_lang[lang]))
    while any(by_lang.values()):
        for lang in order:
            if by_lang[lang]:
                out.append(by_lang[lang].pop(0))
    return out[: req.limit]
