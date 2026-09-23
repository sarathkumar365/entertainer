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


class ModelNotReady(EntertainerError):
    """A build artefact this request needs is absent or out of date.

    Distinct from a bug and from an empty catalogue: the pipeline simply has
    not produced this piece yet. The right response is to finish the build,
    so callers translate it into "come back later", not "something broke".
    """


class CatalogueBusy(EntertainerError):
    """Another process holds the database lock, almost always a build.

    DuckDB gives a writer exclusive access to the file. That is a normal
    state of the system rather than a failure, and a reader that meets it
    should say so rather than raise a driver-level IO error.
    """
