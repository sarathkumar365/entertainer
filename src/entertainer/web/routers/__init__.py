"""Route groups.

Order matters when they are included: /api/validation/summary must be
registered before anything shaped like /api/validation/{case_id}, or the
literal path is swallowed by the parameterised one.
"""

from __future__ import annotations

from . import additions, catalogue, pages, slates, validation, verdicts

#: Include order, mirroring the order the routes were declared in when they
#: all lived in create_app.
ALL = (pages, catalogue, verdicts, validation, slates, additions)

__all__ = ["ALL", "additions", "catalogue", "pages", "slates", "validation", "verdicts"]
