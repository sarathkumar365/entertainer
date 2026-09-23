"""Turn what a person types into a catalogue row.

The entire input surface of this engine is a title typed from memory, so
resolution has to be forgiving in the specific ways human recall fails:
dropped articles, anglicised spellings of non-Latin titles ("Kumbalangi
Nights" vs "Kumbalanki"), the wrong year, the English title when the
catalogue holds the original, and the reverse.

Strategy is a cascade, cheapest first, stopping as soon as a tier produces an
unambiguous winner. Ambiguity is surfaced to the caller rather than guessed
at, because silently recording a verdict against the wrong film poisons the
training set in a way that is very hard to notice later.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import duckdb

_YEAR_RE = re.compile(r"[\s(\[]+((?:19|20)\d{2})[\s)\]]*$")
_ARTICLES = ("the ", "a ", "an ")


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    for art in _ARTICLES:
        if text.startswith(art):
            text = text[len(art):]
            break
    return text


def split_year(query: str) -> tuple[str, int | None]:
    m = _YEAR_RE.search(query.strip())
    if not m:
        return query.strip(), None
    return query[: m.start()].strip(), int(m.group(1))


@dataclass
class Match:
    item_id: int
    title: str
    original_title: str | None
    year: int | None
    kind: str
    language: str | None
    imdb_votes: int | None
    imdb_rating: float | None
    score: float

    def label(self) -> str:
        bits = [self.title]
        if self.original_title and self.original_title != self.title:
            bits.append(f"({self.original_title})")
        tail = []
        if self.year:
            tail.append(str(self.year))
        if self.language:
            tail.append(self.language)
        if self.kind == "tv":
            tail.append("series")
        return " ".join(bits) + (f" [{', '.join(tail)}]" if tail else "")


_SELECT = """
    SELECT item_id, title, original_title, year, kind, language, imdb_votes, imdb_rating
"""


def _rows_to_matches(rows, score_fn) -> list[Match]:
    out = []
    for r in rows:
        out.append(
            Match(
                item_id=int(r[0]), title=r[1], original_title=r[2], year=r[3],
                kind=r[4], language=r[5], imdb_votes=r[6], imdb_rating=r[7],
                score=score_fn(r),
            )
        )
    return out


def _popularity_bonus(votes: int | None) -> float:
    """A tiebreak, not a ranking signal.

    Among titles whose names match equally well, the one with two hundred
    thousand votes is overwhelmingly more likely to be the one being typed
    than an obscure namesake. Capped low so it can never override a better
    textual match.
    """
    import math

    return 0.06 * min(1.0, math.log1p(votes or 0) / math.log1p(500_000))


def search(
    con: duckdb.DuckDBPyConnection,
    query: str,
    limit: int = 8,
    kind: str | None = None,
) -> list[Match]:
    raw, year = split_year(query)
    norm = normalise(raw)
    if not norm:
        return []

    kind_filter = kind if kind in ("movie", "tv") else None
    kind_clause = " AND kind = ?" if kind_filter else ""
    kind_param = [kind_filter] if kind_filter else []
    year_clause = ""
    params: list = []
    if year:
        # Release-year memory is routinely off by one; allow a small window
        # and let the scorer prefer the exact hit.
        year_clause = " AND (year BETWEEN ? AND ?)"
        params += [year - 1, year + 1]

    def year_bonus(r) -> float:
        if not year or not r[3]:
            return 0.0
        return 0.10 if r[3] == year else 0.03

    # Tier 1: exact normalised match on either title form.
    rows = con.execute(
        f"""{_SELECT}
        FROM titles
        WHERE (lower(regexp_replace(title, '[^a-zA-Z0-9 ]', ' ', 'g')) = ?
            OR lower(regexp_replace(original_title, '[^a-zA-Z0-9 ]', ' ', 'g')) = ?
            OR lower(title) = ? OR lower(original_title) = ?)
        {kind_clause}{year_clause}
        ORDER BY imdb_votes DESC NULLS LAST LIMIT 40
        """,
        [norm, norm, raw.lower(), raw.lower(), *kind_param, *params],
    ).fetchall()
    matches = _rows_to_matches(rows, lambda r: 1.0 + _popularity_bonus(r[6]) + year_bonus(r))
    if matches:
        return sorted(matches, key=lambda m: -m.score)[:limit]

    # Tier 2: substring containment, both directions.
    rows = con.execute(
        f"""{_SELECT}
        FROM titles
        WHERE (lower(title) LIKE ? OR lower(original_title) LIKE ?)
        {kind_clause}{year_clause}
        ORDER BY imdb_votes DESC NULLS LAST LIMIT 120
        """,
        [f"%{raw.lower()}%", f"%{raw.lower()}%", *kind_param, *params],
    ).fetchall()

    def contain_score(r) -> float:
        best = 0.0
        for cand in (r[1], r[2]):
            if not cand:
                continue
            c = normalise(cand)
            if not c:
                continue
            # Reward matches that cover most of the candidate title, so
            # "Drishyam" does not lose to "Drishyam 2: The Resumption".
            if norm in c:
                best = max(best, 0.55 + 0.30 * (len(norm) / len(c)))
        return best + _popularity_bonus(r[6]) + year_bonus(r)

    matches = [m for m in _rows_to_matches(rows, contain_score) if m.score > 0]
    if matches:
        top = sorted(matches, key=lambda m: -m.score)
        if top[0].score >= 0.80:
            return top[:limit]

    # Tier 3: fuzzy. Skipped for short numeric queries: films called "96",
    # "1917" and "12" all exist, and character-level similarity between short
    # digit strings is noise — it would resolve "99" to "96" and record a
    # verdict against a film the user never mentioned.
    if norm.isdigit() and len(norm) <= 4:
        return sorted(matches, key=lambda m: -m.score)[:limit]

    # Restricted to a candidate pool by shared first token, because
    # Jaro-Winkler over 300k rows per keystroke is not free.
    first = norm.split(" ")[0][:4]
    rows = con.execute(
        f"""{_SELECT}
        FROM titles
        WHERE (lower(title) LIKE ? OR lower(original_title) LIKE ?)
        {kind_clause}
        ORDER BY imdb_votes DESC NULLS LAST LIMIT 4000
        """,
        [f"{first}%", f"{first}%", *kind_param],
    ).fetchall()
    if not rows:
        where = "WHERE kind = ?" if kind_filter else ""
        rows = con.execute(
            f"""{_SELECT} FROM titles {where}
            ORDER BY imdb_votes DESC NULLS LAST LIMIT 30000""",
            kind_param,
        ).fetchall()

    try:
        from difflib import SequenceMatcher
    except ImportError:  # pragma: no cover
        return sorted(matches, key=lambda m: -m.score)[:limit]

    def fuzzy_score(r) -> float:
        best = 0.0
        for cand in (r[1], r[2]):
            if not cand:
                continue
            best = max(best, SequenceMatcher(None, norm, normalise(cand)).ratio())
        return best * 0.9 + _popularity_bonus(r[6]) + year_bonus(r)

    fuzzy = _rows_to_matches(rows, fuzzy_score)
    pool = {m.item_id: m for m in matches}
    for m in fuzzy:
        if m.item_id not in pool or m.score > pool[m.item_id].score:
            pool[m.item_id] = m
    ranked = sorted(pool.values(), key=lambda m: -m.score)
    return [m for m in ranked if m.score >= 0.45][:limit]


def resolve_one(
    con: duckdb.DuckDBPyConnection, query: str, kind: str | None = None
) -> tuple[Match | None, list[Match]]:
    """Return (confident match or None, alternatives).

    A match is only confident when it is both good in absolute terms and
    clearly better than the runner-up. Anything else comes back as a
    disambiguation list.
    """
    hits = search(con, query, limit=6, kind=kind)
    if not hits:
        return None, []
    if len(hits) == 1:
        return (hits[0], []) if hits[0].score >= 0.6 else (None, hits)
    top, second = hits[0], hits[1]
    if top.score >= 0.95 and top.score - second.score >= 0.08:
        return top, hits[1:]
    if top.score >= 0.75 and top.score - second.score >= 0.20:
        return top, hits[1:]
    return None, hits


def by_item_id(con: duckdb.DuckDBPyConnection, item_id: int) -> Match | None:
    """Look up a row that is already known by id.

    Same column list as every other lookup here, which is the reason this
    exists: the CLI had its own copy of the SELECT, so adding a column to
    Match meant remembering to edit a query 1,200 lines away in another file.
    """
    row = con.execute(f"{_SELECT} FROM titles WHERE item_id = ?", [int(item_id)]).fetchone()
    if not row:
        return None
    return _rows_to_matches([row], lambda _r: 1.0)[0]


def from_slate(con: duckdb.DuckDBPyConnection, query: str) -> Match | None:
    """Let a bare number refer to a position in the last recommended slate.

    The daily loop is: ask for recommendations, watch one, say what you
    thought. Retyping a title just read off the screen — often a
    transliterated one — is the friction most likely to stop someone giving
    feedback at all, and feedback is the only thing this system runs on.

    Positions are 1-based because that is how the slate is printed. Anything
    out of range returns None rather than raising, so "12" simply falls
    through to being treated as a title.
    """
    from . import store

    text = query.strip()
    if not text.isdigit():
        return None
    position = int(text)
    slate = store.last_slate(con)
    if not (1 <= position <= len(slate)):
        return None
    return by_item_id(con, slate[position - 1])
