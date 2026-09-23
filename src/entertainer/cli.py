"""Command line interface.

The whole interaction model is: type a title, say what you thought.
Everything else the engine does is downstream of that. Commands are
therefore named after what a person would say out loud — ``ent loved
parasite`` — rather than after the machinery underneath.

The commands themselves live in ``commands``, grouped by what they are for.
This module is the entry point named in pyproject and nothing else; it stays
a module rather than becoming a package so that ``git log --follow`` still
works on the file with the most history behind it.
"""

from __future__ import annotations

import sys

from .commands import app
from .render import console

__all__ = ["app", "main"]


def main() -> None:  # pragma: no cover
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[dim]interrupted[/dim]")
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()
