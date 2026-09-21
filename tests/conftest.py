"""Test-wide safety rails.

The suite must never reach the network or touch the developer's real
credentials. Deleting the TMDB environment variables is not enough on its
own, because `config` calls `load_dotenv()` at import and re-reads the real
`.env` — which it did, and one test consequently made a live API call.
"""

from __future__ import annotations

import os

os.environ.setdefault("ENTERTAINER_NO_DOTENV", "1")
os.environ.pop("TMDB_API_KEY", None)
os.environ.pop("TMDB_BEARER", None)
