"""Read a Netflix thumbs history and resolve it against TMDB.

Netflix exports its rating history as a Falcor jsonGraph blob, one page of
fifty at a time. Each item carries a title string, a thumbs verdict, a date,
and an internal `movieID` that maps to nothing outside Netflix — there is no
public crosswalk from it to IMDb or TMDB. So the only usable join key is the
title text, and title text is ambiguous: "Hunger", "Alpha", "Sahara" and
"Youth" each name several unrelated films.

Attaching a verdict to the wrong film is the worst failure this project has
had, so resolution here is explicitly graded rather than best-effort. A match
is only automatic when the normalised title is an exact hit, the release date
precedes the rating date, and nothing else ties it. Everything weaker is
reported for a human to settle.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from . import tmdb

# Netflix's three-way thumbs, mapped onto the project's verdict names. Only
# the name is recorded: the numeric reward belongs to `models.taste.VERDICTS`
# and duplicating it here would let the two drift apart silently.
THUMB_VERDICTS: dict[str, str] = {
    "THUMBS_WAY_UP": "love",
    "THUMBS_UP": "like",
    "THUMBS_DOWN": "dislike",
}

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE = re.compile(r"\s+")
# Apostrophes are elided rather than spaced: Netflix writes a curly one and
# TMDB a straight one, and turning either into a space would split "Pope's"
# into two tokens while the other source keeps one.
_APOSTROPHE = re.compile(r"[\u0027\u2018\u2019\u02bc\u00b4`]")

# Netflix appends edition markers that TMDB's title does not carry.
_EDITION = re.compile(
    r"\s*[:(\-–—]?\s*(reloaded version|extended (cut|version)|director'?s cut|"
    r"based on a true story|limited series)\s*\)?\s*$",
    re.IGNORECASE,
)


def normalise(title: str) -> str:
    """Casefold, strip accents and punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKD", title)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = _APOSTROPHE.sub("", text.casefold())
    text = _PUNCT.sub(" ", text)
    return _SPACE.sub(" ", text).strip()


def _strip_edition(title: str) -> str:
    previous = None
    while previous != title:
        previous = title
        title = _EDITION.sub("", title).strip()
    return title


def parse_date(raw: str) -> date | None:
    """Netflix writes `YY-M-DD`. Two-digit years are 2000s."""
    parts = raw.split("-")
    if len(parts) != 3:
        return None
    try:
        year, month, day = (int(p) for p in parts)
    except ValueError:
        return None
    try:
        return date(2000 + year if year < 100 else year, month, day)
    except ValueError:
        return None


@dataclass
class Rating:
    title: str
    thumbs: str
    netflix_id: int | None
    rated_on: date | None

    @property
    def verdict(self) -> str:
        return THUMB_VERDICTS[self.thumbs]


def parse(payload: dict | str | Path) -> list[Rating]:
    """Accept a jsonGraph blob, a bare `{"ratingItems": [...]}`, or a path.

    Pages are merged by the caller; this reads one blob. Items whose thumbs
    value is unrecognised are dropped rather than guessed at.
    """
    if isinstance(payload, (str, Path)) and Path(payload).exists():
        payload = json.loads(Path(payload).read_text(encoding="utf-8"))
    elif isinstance(payload, str):
        payload = json.loads(payload)

    node = payload
    for key in ("jsonGraph", "aui", "ratingHistory", "value"):
        if isinstance(node, dict) and key in node:
            node = node[key]
    items = (node or {}).get("ratingItems") or []

    out: list[Rating] = []
    for item in items:
        thumbs = item.get("thumbs") or (item.get("reactionRatings") or {}).get("thumbs")
        if thumbs not in THUMB_VERDICTS:
            continue
        out.append(
            Rating(
                title=(item.get("title") or "").strip(),
                thumbs=thumbs,
                netflix_id=item.get("movieID"),
                rated_on=parse_date(item.get("date") or ""),
            )
        )
    return out


@dataclass
class Resolution:
    rating: Rating
    confidence: str  # "high" | "ambiguous" | "low" | "unresolvable"
    reason: str
    match: dict | None = None
    alternatives: list[dict] = field(default_factory=list)

    @property
    def automatic(self) -> bool:
        return self.confidence == "high"


def _year(hit: dict) -> int | None:
    raw = hit.get("release_date") or hit.get("first_air_date") or ""
    return int(raw[:4]) if raw[:4].isdigit() else None


def _candidate(hit: dict) -> dict:
    return {
        "tmdb_id": hit.get("id"),
        "kind": hit.get("_kind"),
        "title": hit.get("title") or hit.get("name"),
        "original_title": hit.get("original_title") or hit.get("original_name"),
        "year": _year(hit),
        "language": hit.get("original_language"),
        "popularity": round(hit.get("popularity") or 0.0, 1),
        "votes": hit.get("vote_count") or 0,
    }


def resolve_one(rating: Rating) -> Resolution:
    """Grade one Netflix row against TMDB search."""
    if not rating.title:
        return Resolution(
            rating, "unresolvable",
            "Netflix exported an empty title; only its internal id remains",
        )

    query = _strip_edition(rating.title)
    hits = tmdb.search(query)
    if not hits:
        return Resolution(rating, "low", "TMDB search returned nothing")

    wanted = normalise(query)
    candidates = [_candidate(h) for h in hits]

    # A film cannot be rated before it exists, so the rating date is a hard
    # upper bound on the release year. It is a weak filter — one year of
    # slack for regional release lag — but it removes the worst collisions,
    # where a recent remake outranks the film actually watched.
    if rating.rated_on:
        bound = rating.rated_on.year
        dated = [c for c in candidates if c["year"] is None or c["year"] <= bound]
        if dated:
            candidates = dated

    exact = [
        c for c in candidates
        if normalise(c["title"] or "") == wanted
        or normalise(c["original_title"] or "") == wanted
    ]

    if not exact:
        return Resolution(
            rating, "low",
            f"no exact title match for {query!r}",
            match=candidates[0], alternatives=candidates[1:4],
        )

    best = exact[0]
    rest = exact[1:]
    if not rest:
        return Resolution(rating, "high", "unique exact title match", match=best)

    # Several films share the title. Netflix's own catalogue is thin enough
    # that popularity is usually right, but "usually" is what caused the
    # earlier mis-attachment, so this is handed to a human instead.
    runner = rest[0]
    margin = (best["popularity"] + 1.0) / (runner["popularity"] + 1.0)
    if margin >= 8.0 and best["votes"] > runner["votes"]:
        return Resolution(
            rating, "high",
            f"exact match {margin:.0f}x more popular than next",
            match=best, alternatives=rest[:3],
        )
    return Resolution(
        rating, "ambiguous",
        f"{len(exact)} films share this title",
        match=best, alternatives=rest[:3],
    )


def resolve(ratings: list[Rating], *, progress=None) -> list[Resolution]:
    out = []
    for index, rating in enumerate(ratings, 1):
        out.append(resolve_one(rating))
        if progress:
            progress(index, len(ratings))
    return out


def flag_collisions(resolutions: list[Resolution]) -> list[Resolution]:
    """Demote matches where two Netflix rows resolved to the same film.

    Netflix carries remakes under identical titles — the Chinese and Korean
    "A Love So Beautiful" are separate entries with opposite verdicts. Text
    search cannot tell them apart, and writing both would put contradictory
    verdicts on one catalogue row, which is worse than importing neither.
    """
    counts: dict[tuple[int, str], int] = {}
    for res in resolutions:
        if res.match and res.match.get("tmdb_id"):
            key = (int(res.match["tmdb_id"]), str(res.match.get("kind")))
            counts[key] = counts.get(key, 0) + 1

    for res in resolutions:
        if not res.match or not res.match.get("tmdb_id"):
            continue
        key = (int(res.match["tmdb_id"]), str(res.match.get("kind")))
        if counts[key] > 1:
            res.confidence = "ambiguous"
            res.reason = (
                f"{counts[key]} Netflix rows resolved to this same film; "
                "likely a remake sharing its title"
            )
    return resolutions


def apply_decisions(
    resolutions: list[Resolution], decisions: dict[str, object]
) -> list[Resolution]:
    """Fold human rulings into a resolved set, keyed by Netflix id.

    Text matching cannot settle which "Hunger" someone watched, so the
    ambiguous cases are handed to a person once and their answers recorded
    here. Keeping them in a file rather than in a one-off script means a
    later page of the same export re-imports identically.

    A ruling is "accept" (take the best guess), "skip" (never import), or an
    explicit `{"tmdb_id": ..., "kind": ...}`.
    """
    for res in resolutions:
        ruling = decisions.get(str(res.rating.netflix_id))
        if ruling is None:
            continue
        if ruling == "skip":
            res.confidence = "skipped"
            res.reason = "ruled out by hand"
            continue
        if ruling == "accept":
            if not res.match:
                res.confidence = "low"
                res.reason = "accepted by hand, but there was no candidate to accept"
                continue
            res.confidence = "high"
            res.reason = "best guess confirmed by hand"
            continue
        if isinstance(ruling, dict) and ruling.get("tmdb_id"):
            res.match = {
                "tmdb_id": int(ruling["tmdb_id"]),
                "kind": ruling.get("kind", "movie"),
                "title": ruling.get("title") or res.rating.title,
                "year": ruling.get("year"),
                "language": ruling.get("language"),
            }
            res.confidence = "high"
            res.reason = "chosen by hand"
            continue
        raise ValueError(f"unrecognised ruling for {res.rating.netflix_id}: {ruling!r}")
    return resolutions


def dedupe(ratings: list[Rating]) -> list[Rating]:
    """Drop repeats, keeping the most recent verdict for each title.

    Netflix pages overlap when they are grabbed by hand, and later pages hold
    older verdicts, so the first occurrence of a title is the most recent
    opinion. Keyed on the normalised title *and* the Netflix id, because two
    genuinely different titles can normalise alike and a single title can be
    exported under more than one id.
    """
    seen: set[tuple[str, int | None]] = set()
    unique: list[Rating] = []
    for rating in ratings:
        key = (normalise(rating.title), rating.netflix_id)
        if key in seen:
            continue
        seen.add(key)
        unique.append(rating)
    return unique


def review_payload(resolutions: list[Resolution]) -> list[dict]:
    """What a human needs in order to settle an ambiguous title.

    The Netflix id is carried through because the decisions file is keyed on
    it: rulings recorded once then apply to every later re-import of the same
    export, rather than being retyped.
    """
    return [
        {
            "title": r.rating.title,
            "thumbs": r.rating.thumbs,
            "verdict": r.rating.verdict,
            "netflix_id": r.rating.netflix_id,
            "rated_on": str(r.rating.rated_on) if r.rating.rated_on else None,
            "confidence": r.confidence,
            "reason": r.reason,
            "best_guess": r.match,
            "alternatives": r.alternatives,
        }
        for r in resolutions
    ]


def bucket(resolutions: list[Resolution]) -> dict[str, list[Resolution]]:
    """Group resolutions by confidence."""
    out: dict[str, list[Resolution]] = {}
    for resolution in resolutions:
        out.setdefault(resolution.confidence, []).append(resolution)
    return out


def clear_previous(con) -> int:
    """Remove verdicts from an earlier Netflix import.

    Re-running after settling the ambiguous titles would otherwise stack a
    second verdict on every title the first pass already wrote. The latest
    verdict wins so nothing breaks, but the duplicates distort the
    prequential replay, which walks the log in order.
    """
    cleared = con.execute("SELECT count(*) FROM events WHERE source = 'netflix'").fetchone()[0]
    con.execute("DELETE FROM events WHERE source = 'netflix'")
    return int(cleared)
