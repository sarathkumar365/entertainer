"""Turn a catalogue row into the text the encoder actually sees.

The card is deliberately written as natural prose rather than a key-value dump.
Sentence encoders are trained on prose, and the difference in downstream
retrieval quality between "genres: Drama|Crime" and "A crime drama about…" is
large.

Ordering matters too. The synopsis goes last and is given the most room,
because it carries the signal that genre labels cannot: pacing, tone, moral
posture, whether the ending resolves. Those are the things that actually decide
whether someone likes a film, and none of them appear in a tag vocabulary.
"""

from __future__ import annotations

from ..config import PRIORITY_LANGUAGES

_MAX_OVERVIEW = 900


def _people(row: dict, key: str, limit: int) -> list[str]:
    vals = row.get(key) or []
    return [v for v in vals if v][:limit]


def language_name(code: str | None) -> str:
    if not code:
        return ""
    return PRIORITY_LANGUAGES.get(code, code)


def build_card(row: dict) -> str:
    title = row.get("title") or row.get("original_title") or "Untitled"
    original = row.get("original_title")
    year = row.get("year")
    kind = "series" if row.get("kind") == "tv" else "film"
    lang = language_name(row.get("language"))
    runtime = row.get("runtime")

    head = title
    if original and original != title:
        head += f" ({original})"
    bits = []
    if year:
        bits.append(str(year))
    if lang:
        bits.append(f"{lang}-language {kind}")
    else:
        bits.append(kind)
    if runtime:
        bits.append(f"{runtime} min")
    head += " — " + ", ".join(bits) + "."

    lines = [head]

    directors = _people(row, "directors", 2)
    cast = _people(row, "cast_names", 5)
    credits = []
    if directors:
        credits.append("Directed by " + " and ".join(directors))
    if cast:
        credits.append("starring " + ", ".join(cast))
    if credits:
        lines.append(". ".join(credits) + ".")

    genres = [g for g in (row.get("genres") or []) if g]
    if genres:
        lines.append("Genre: " + ", ".join(dict.fromkeys(genres)) + ".")

    keywords = [k for k in (row.get("keywords") or []) if k]
    if keywords:
        lines.append("Themes: " + ", ".join(dict.fromkeys(keywords[:18])) + ".")

    tagline = (row.get("tagline") or "").strip()
    if tagline:
        lines.append(tagline.rstrip(".") + ".")

    overview = (row.get("overview") or "").strip()
    if overview:
        lines.append(overview[:_MAX_OVERVIEW])

    return "\n".join(lines)


QUERY_INSTRUCTION = (
    "Given a description of someone's taste in film and television, "
    "retrieve titles that would satisfy it."
)
