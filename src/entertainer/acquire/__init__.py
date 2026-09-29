"""Getting a judged title onto disk, ready for Jellyfin.

Spike. Sources are adapters behind one small protocol so that the decision of
*where* a title comes from stays an explicit configuration choice rather than
something inherited from whichever library was convenient (README, Part two).
This package is a capability: it takes a resolved ``Target`` and never looks
titles up itself, so it imports nothing local but ``config`` and ``errors``.
"""

from __future__ import annotations

from .core import Acquired, acquire
from .match import Candidate, Target

__all__ = ["Acquired", "Candidate", "Target", "acquire"]
