"""The Typer applications every command group decorates.

Kept apart from the command modules so that they can all import the same
objects without importing each other. ``cli`` imports ``commands`` for its
side effects — a command exists only once its module has been imported and
its decorator has run.
"""

from __future__ import annotations

import typer

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="A personal, self-improving recommendation engine for film and television.",
)

data_app = typer.Typer(no_args_is_help=True, help="Build and maintain the catalogue.")
bundle_app = typer.Typer(no_args_is_help=True, help="Move a built catalogue between machines.")

# Registration order decides the order `ent --help` lists the groups.
app.add_typer(data_app, name="data")
app.add_typer(bundle_app, name="bundle")
