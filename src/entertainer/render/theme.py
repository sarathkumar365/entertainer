"""The console, and the formatting primitives shared across commands."""

from __future__ import annotations

import typer
from rich.console import Console

# One console for the whole CLI. Deliberately constructed without a width:
# rich then reads COLUMNS, so output adapts to the terminal. Tests pin
# COLUMNS instead of pinning a width here.
console = Console()


def fail(message: str) -> None:
    """Print in red and exit non-zero.

    Lives in the render layer because it raises ``typer.Exit``. Library code
    raises an ``EntertainerError`` instead and lets the command translate it.
    """
    console.print(f"[red]{message}[/red]")
    raise typer.Exit(code=1)


def as_ten(value: float) -> float:
    """The 0-10 display scale. See models.taste.to_display_scale."""
    from ..models.taste import to_display_scale

    return to_display_scale(value)


def pm(std: float) -> float:
    """The ± half-width shown beside a prediction.

    Capped at 10 for the same reason as ``as_ten``: a posterior that knows
    almost nothing produces a band wider than the scale it is drawn on.
    """
    return float(min(std * 10, 10.0))


#: Tone words from the library layer, mapped to rich styles. Library code
#: returns a word rather than markup so the same reading can be printed here,
#: serialised as JSON, or rendered in a browser.
TONE = {"good": "green", "bad": "red", "warn": "yellow", "dim": "dim"}


def toned(text: str, tone: str) -> str:
    """Wrap ``text`` in the style for ``tone``, or leave it plain."""
    style = TONE.get(tone)
    return f"[{style}]{text}[/{style}]" if style else text
