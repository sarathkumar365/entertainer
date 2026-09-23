"""Reading and writing the verdict history as a file.

Three file formats lived inside command bodies: the JSONL export, its reader,
and the plain-text bulk list. They are the only durable record of a taste
profile that survives a catalogue rebuild, so they belong somewhere they can
be tested and reused rather than inside a Typer callback.

Nothing here touches the console or typer. Failures raise EntertainerError.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .errors import EntertainerError

DEFAULT_EXPORT = Path("profile.jsonl")


@dataclass
class BulkLine:
    """One line of a bulk import list."""

    title: str
    verdict: str


@dataclass
class ImportCounts:
    """What an import did. Three outcomes, deliberately kept apart.

    ``duplicates`` is not an error — re-importing the same file must be safe —
    and ``missing`` is not either, since a pruned catalogue legitimately no
    longer contains some titles.
    """

    imported: int = 0
    duplicates: int = 0
    missing: int = 0


def parse_bulk_lines(text: str, default_verdict: str = "like") -> list[BulkLine]:
    """Parse a bulk list: one title per line, optional ``| verdict``.

    ``rsplit`` on the last pipe rather than ``split`` on the first, because a
    title may legitimately contain one — "Tinker Tailor Soldier | Spy" would
    otherwise resolve to "Tinker Tailor Soldier".

    Blank lines and ``#`` comments are skipped.
    """
    out: list[BulkLine] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        verdict = default_verdict
        if "|" in line:
            line, verdict = (part.strip() for part in line.rsplit("|", 1))
        # Deliberately not skipping an empty title here. A line of "| love"
        # currently reaches the resolver and is reported as unresolved, and
        # this extraction is meant to be behaviour-preserving; tightening it
        # is a separate decision.
        out.append(BulkLine(title=line, verdict=verdict))
    return out


def export_events(con, path: Path) -> int:
    """Write every verdict as JSONL, joined to the title it refers to.

    Keyed on ``imdb_id`` rather than the internal ``item_id``, which is a
    row number that a rebuild reassigns. That is the whole point of the
    format: it survives the catalogue being thrown away.
    """
    rows = con.execute(
        """
        SELECT t.imdb_id, t.title, t.year, e.kind, e.value, e.source, e.ts, e.context
        FROM events e JOIN titles t USING (item_id) ORDER BY e.ts
        """
    ).fetchall()
    with open(path, "w", encoding="utf-8") as fh:
        for imdb_id, title, year, kind, value, source, ts, context in rows:
            fh.write(
                json.dumps(
                    {
                        "imdb_id": imdb_id,
                        "title": title,
                        "year": year,
                        "kind": kind,
                        "value": value,
                        "source": source,
                        "ts": str(ts),
                        "context": json.loads(context or "{}"),
                    }
                )
                + "\n"
            )
    return len(rows)


def import_events(con, path: Path) -> ImportCounts:
    """Re-import an exported profile, matching on IMDb id.

    Idempotent through ``store.import_event``, so running it twice does not
    double-count.
    """
    from . import store

    counts = ImportCounts()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EntertainerError(f"{path}:{number} is not valid JSON: {exc}") from exc

        row = con.execute(
            "SELECT item_id FROM titles WHERE imdb_id = ?", [record.get("imdb_id")]
        ).fetchone()
        if not row:
            counts.missing += 1
            continue
        try:
            added = store.import_event(
                con,
                int(row[0]),
                record["kind"],
                record.get("value"),
                record.get("source", "import"),
                record["ts"],
                record.get("context"),
            )
        except (KeyError, ValueError) as exc:
            raise EntertainerError(f"{path}:{number} is not a valid profile record: {exc}") from exc
        if added:
            counts.imported += 1
        else:
            counts.duplicates += 1
    return counts
