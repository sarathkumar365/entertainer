"""Errors whose message is safe to show a person.

Library code must not import typer. A module that raises ``typer.Exit`` can
only be called from a command, which is how business rules ended up reachable
only by invoking a Typer callback in the first place. Raising a domain error
instead lets the same function serve the CLI, the web app and a test.

``IngestError`` already set this precedent; it is now part of the family.

The CLI translates these at the boundary — see ``commands._shared``.
"""

from __future__ import annotations


class EntertainerError(RuntimeError):
    """Base class. The message is written for a person, not a stack trace."""


class MissingCatalogue(EntertainerError):
    """The catalogue has not been built yet."""


class NotEnoughEvidence(EntertainerError):
    """A measurement was asked for before there was enough data to make it."""


class IntegrityRefusal(EntertainerError):
    """An evaluation would have been misleading, so it was refused.

    Distinct from a bug: the code worked, and declined to produce a number
    that could not be trusted.
    """
