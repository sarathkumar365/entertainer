"""Terminal rendering.

Kept apart from the commands so that the logic underneath can be called
without a console attached. Anything in here may import rich and typer;
nothing outside may.
"""

from __future__ import annotations

from .theme import as_ten, console, fail, pm

__all__ = ["as_ten", "console", "fail", "pm"]
