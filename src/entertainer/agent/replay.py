"""Re-running the matcher and the decision rule over every recorded job, offline.

The merge gate for matcher changes (``docs/agent/DESIGN.md`` phase 7): a change
that raises the wrong-pick count does not merge. Each job file already holds
the full candidate list and the film it was matched against, so a replay needs
no Telegram connection and no TMDB call, and every real request becomes a test
case without anyone writing one.

The labels come from the user, who is the only judge of "the right file":

- a copy filed and never called wrong is right;
- a copy chosen at a pause is right, because the user said so;
- a copy called "wrong file" is wrong;
- a dry run confirmed with ``ent acquire confirm`` names its right file, which is
  how the seed set of trap films gets labels without downloading anything.

A pick with no label either way is counted as unlabelled, not as right: the
replay must not flatter a change by assuming the unknown went well.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..acquire import Candidate, Plan, Target
from ..acquire.match import score
from ..config import PATHS, acquire_settings
from ..errors import NotAcquired
from . import jobs as J
from .decide import decide


@dataclass
class Case:
    job: str
    folder: str
    outcome: str  # right | wrong | paused | unlabelled | nothing
    picked: str = ""
    why: str = ""


@dataclass
class Replay:
    cases: list[Case] = field(default_factory=list)

    def count(self, outcome: str) -> int:
        return sum(1 for c in self.cases if c.outcome == outcome)

    @property
    def totals(self) -> dict[str, int]:
        return {k: self.count(k) for k in ("right", "wrong", "paused", "unlabelled", "nothing")}


def baseline_path() -> Path:
    return PATHS.root / "acquire" / "replay-baseline.json"


def labels(job: J.Job) -> tuple[set[str], set[str]]:
    right: set[str] = set()
    if job.filed:
        right.add(job.filed["label"])
    if job.choice:
        right.add(job.choice["label"])
    if job.confirmed:
        right.add(job.confirmed)
    wrong = set(job.wrong)
    return right - wrong, wrong


def replay(jobs: J.Jobs | None = None, max_gb: float | None = None) -> Replay:
    cap = max_gb if max_gb is not None else acquire_settings().max_gb
    out = Replay()
    for job in sorted((jobs or J.Jobs()).all(), key=lambda j: j.id):
        if not (job.target and job.candidates):
            continue
        target = Target(**{**job.target, "alt_titles": tuple(job.target.get("alt_titles") or ())})
        cands = [score(target, Candidate(c["source"], c["label"], c.get("size"), None,
                                         ext=_ext(c)))
                 for c in job.candidates]
        cands.sort(key=lambda c: -c.score)
        right, wrong = labels(job)
        try:
            d = decide(Plan(target, cands), cap)
        except NotAcquired as e:
            out.cases.append(Case(job.id, job.folder or "", "nothing", why=str(e)[:120]))
            continue
        if d.paused:
            out.cases.append(Case(job.id, job.folder or "", "paused", why=d.reason))
            continue
        top = d.picks[0].label
        outcome = "right" if top in right else "wrong" if top in wrong else "unlabelled"
        out.cases.append(Case(job.id, job.folder or "", outcome, picked=top, why=d.reason))
    return out


def _ext(c: dict) -> str:
    # The extension a bot's delivered file revealed is part of the record, but a
    # replay must judge the label as the matcher first saw it.
    return c.get("ext") or ""


def save_baseline(r: Replay) -> Path:
    path = baseline_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"totals": r.totals, "cases": [asdict(c) for c in r.cases]},
                               indent=2))
    return path


def load_baseline() -> dict | None:
    path = baseline_path()
    return json.loads(path.read_text()) if path.is_file() else None
