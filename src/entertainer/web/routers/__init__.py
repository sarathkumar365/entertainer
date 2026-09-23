"""Route groups.

Order matters when they are included. /api/validation/summary must be
registered before anything shaped like /api/validation/{case_id}, or the
literal path is swallowed by the parameterised one — and `pages` must be
last, because it ends with a catch-all that would otherwise swallow
everything.
"""

from __future__ import annotations

from . import additions, catalogue, insight, library, pages, slates, validation, verdicts

#: Include order, mirroring the order the routes were declared in when they
#: all lived in create_app.
ALL = (catalogue, verdicts, library, validation, slates, insight, additions, pages)

__all__ = [
    "ALL", "additions", "catalogue", "insight", "library", "pages",
    "slates", "validation", "verdicts",
]
