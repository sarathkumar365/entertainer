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
    """Render a reward on a 0-10 scale a person can read.

    The posterior is an unbounded linear model, so it will happily predict
    10.4 for something squarely in the middle of what you love. That is
    correct arithmetic and nonsense as a displayed score, so it is clamped —
    at the display layer only. Clamping the model itself would distort the
    ranking and throw away the information that one title is further along
    the preference direction than another.
    """
    return float(min(10.0, max(0.0, value * 10.0)))


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
