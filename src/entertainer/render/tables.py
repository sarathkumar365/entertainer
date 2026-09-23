"""Tables that more than one command draws."""

from __future__ import annotations

from collections.abc import Iterable

from rich.table import Table

from ..config import language_label


def language_histogram(rows: Iterable[tuple[str, int]]) -> Table:
    """Language against title count.

    Drawn identically by `ent data build`, `ent data prune` and `ent stats`;
    it was written out three times. Callers supply the rows, because they come
    from different queries with different limits.
    """
    table = Table("language", "titles")
    for language, count in rows:
        table.add_row(f"{language} ({language_label(language)})", f"{count:,}")
    return table
