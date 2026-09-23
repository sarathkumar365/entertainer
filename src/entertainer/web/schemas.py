"""Request bodies."""

from __future__ import annotations

from pydantic import BaseModel, Field

VERDICT_HELP = "love | like | ok | meh | dislike | hate | unseen"


class Verdict(BaseModel):
    item_id: int
    verdict: str = Field(description=VERDICT_HELP)
    #: Set when the verdict came from a recommendation slate. Stamped into
    #: the event context so the off-policy join is exact rather than
    #: temporal — without it a rating is credited to a slate only because it
    #: happened to come later.
    slate_id: str | None = None
    position: int | None = None


class AddRequest(BaseModel):
    tmdb_id: int
    kind: str = "movie"
    verdict: str | None = None


class ValidationSealRequest(BaseModel):
    #: At least twenty, or the paired comparison has no power; at most a
    #: hundred, because every sealed title is one you cannot rate until you
    #: reveal it.
    item_ids: list[int] = Field(min_length=20, max_length=100)


class ValidationRevealRequest(BaseModel):
    verdict: str = Field(description=VERDICT_HELP)
