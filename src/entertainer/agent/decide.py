"""Which file to take, or whether to stop and ask.

The user has already decided *whether* to get the film; this decides *which
copy*, by the table in ``docs/agent/DESIGN.md`` phase 4. Picking is deferred
rather than approved: most picks are unambiguous, and asking every time would
make the user the bottleneck for the few that are not. So the rule pauses only
when the evidence is thin (a low score, or a label silent on the year or on the
film's language) or contradicts itself (the leading copies name different
films, years or languages).

"Leading" means within ``CLOSE`` of the best score. A year or a language costs
three points either way (+2 against -1), so a copy that disagrees on either and
still sits within three points of the best has won on quality alone, which is
exactly the case the matcher cannot settle. A dub further down is the matcher
doing its job, not a conflict, and pausing for it would pause on every Tamil
film, since request groups index every dub under the original title.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..acquire import Candidate, Plan, Target
from ..acquire.core import GB
from ..errors import NotAcquired

AUTO_PICK = 13.0
CLOSE = 3.0
OFFER = 3


@dataclass
class Decision:
    """``picks`` is the order to try in when nobody is asked; ``offer`` is what a
    pause shows. Exactly one of them is filled."""

    picks: list[Candidate] = field(default_factory=list)
    offer: list[Candidate] = field(default_factory=list)
    reason: str = ""
    over_cap: bool = False

    @property
    def paused(self) -> bool:
        return not self.picks


def decide(p: Plan, max_gb: float) -> Decision:
    usable = p.usable
    if not usable:
        refused = "; ".join(sorted({c.why for c in p.candidates})[:5]) or "nothing found"
        failed = "".join(f" {n} failed ({e})." for n, e in p.errors.items())
        raise NotAcquired(f"No source had {p.target.folder}. Refused: {refused}.{failed}")

    # Unknown size counts as fitting here; execute() stops it mid-download if it does not.
    fits = [c for c in usable if not (c.size and c.size > max_gb * GB)]
    if not fits:
        smallest = min(usable, key=lambda c: c.size or 0)
        return Decision(offer=[smallest], over_cap=True, reason=(
            f"every copy is over the {max_gb:g} GB cap; the smallest is "
            f"{(smallest.size or 0) / GB:.2f} GB"))

    top = fits[0]
    if top.score < AUTO_PICK:
        return Decision(offer=fits[:OFFER], reason=(
            f"the best match scores {top.score:.1f}, under {AUTO_PICK:g}"))
    # The design expected a missing year or language to sink the score under 13,
    # but each costs only three points and a 1080p HEVC copy has four to spare.
    # So both are asked about directly, which is what the table meant.
    if silent := _silent(top, p.target):
        return Decision(offer=fits[:OFFER], reason=f"the best match {silent}")

    leading = [c for c in fits if c.score >= top.score - CLOSE]
    if clash := _disagreement(leading):
        return Decision(offer=leading[:OFFER], reason=f"the closest copies disagree on {clash}")

    # Fall back only to copies that agree with the pick: if the best one fails to
    # verify, the next try must not be the dub the rule would have paused over.
    same = [c for c in fits if not _disagreement([top, c])]
    return Decision(picks=same, reason=f"{top.label} scores {top.score:.1f}")


def _silent(c: Candidate, target: Target) -> str:
    if target.year and c.year is None:
        return "does not say its year"
    if target.language and c.lang != target.language.lower():
        return f"is not marked {target.language}"
    return ""


def _disagreement(cands: list[Candidate]) -> str:
    """Name the first thing the copies disagree on; a label that is silent agrees."""
    for what, key in (("title", "title"), ("year", "year"), ("language", "lang")):
        said = {getattr(c, key) for c in cands} - {None, ""}
        if len(said) > 1:
            return what
    return ""
