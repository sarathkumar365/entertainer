"""The shared table renderers.

`ent data build` and `ent data prune` have no CLI-level coverage, so the
language histogram had two of its three call sites unverified when it was
extracted. These test the renderer and the query underneath it directly.
"""

from __future__ import annotations

import pytest
from rich.console import Console
from test_cli import app_env  # noqa: F401

from entertainer.render import tables


def render(renderable, width: int = 60) -> str:
    console = Console(width=width, file=None, record=True, no_color=True)
    console.print(renderable)
    return console.export_text()


def test_language_histogram_names_and_formats_each_row():
    out = render(tables.language_histogram([("ml", 1234), ("ta", 7)]))
    assert "ml (Malayalam)" in out
    assert "1,234" in out
    assert "ta (Tamil)" in out


def test_language_histogram_keeps_the_order_it_is_given():
    """Ordering is the query's job, not the renderer's — the renderer must not
    re-sort, or the deterministic tiebreaker upstream would be pointless."""
    out = render(tables.language_histogram([("ta", 1), ("ml", 9)]))
    assert out.index("ta (Tamil)") < out.index("ml (Malayalam)")


def test_language_histogram_handles_an_unknown_code():
    out = render(tables.language_histogram([("zz", 3)]))
    assert "zz" in out


@pytest.mark.usefixtures("app_env")
def test_catalogue_histogram_breaks_ties_deterministically():
    """Two languages on the same count must come back in a stable order.

    Without the tiebreaker the same catalogue printed a different table
    between runs, which is how this was found.
    """
    from entertainer.data import catalog

    first = catalog.language_histogram(25)
    for _ in range(4):
        assert catalog.language_histogram(25) == first

    counts = [c for _, c in first]
    assert counts == sorted(counts, reverse=True)
    for i in range(len(first) - 1):
        if first[i][1] == first[i + 1][1]:
            assert first[i][0] < first[i + 1][0], "ties are not ordered by language"
