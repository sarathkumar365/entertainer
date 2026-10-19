"""Getting a resolved title onto disk: sources, matching, verification.

A capability, not the agent. It takes a ``Target`` that someone else resolved,
searches the configured sources, and downloads and verifies what it is told
to; it never looks titles up, never decides on the user's behalf and never
writes to the library or the store. That is ``agent``'s work. It imports
nothing local but ``config`` and ``errors``.
"""

from __future__ import annotations

from .core import Fetched, Plan, execute, plan
from .match import Candidate, Target
from .sources import KNOWN, Source, open_sources

__all__ = [
    "KNOWN", "Candidate", "Fetched", "Plan", "Source", "Target", "execute", "open_sources", "plan",
]
