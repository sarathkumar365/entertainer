"""The agent: turns "get me this film" into a file Jellyfin can play.

The work layer over ``acquire``. It resolves what the user asked for into a
``Target`` by IMDb id, checks this machine can acquire at all, runs the search
and the download, files the result into the library and tells Jellyfin. Jobs and
the worker thread arrive here in later phases
(``docs/agent/ARCHITECTURE.md``).
"""

from __future__ import annotations

from .decide import Decision, decide
from .run import Acquired, get, ready, search
from .target import from_imdb, from_item

__all__ = ["Acquired", "Decision", "decide", "from_imdb", "from_item", "get", "ready", "search"]
