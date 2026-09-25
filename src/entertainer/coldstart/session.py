"""The cold-start questioning loop, with the terminal taken out of it.

`ent onboard` held all of this inside its command body: which titles to ask
about, when to refit, when to fetch a new batch, and what counts as an
answer. None of it is about a terminal, and the web interface needs the same
sequence to offer adaptive questions.

What stays outside: reading a keystroke, and printing. The session is asked
for a question and told an answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..models.taste import fit as fit_taste
from . import elicit

#: Verdicts needed before the adaptive criterion has anything to work with.
#: Below this the questions come from the seeded opening ladder instead.
MIN_FOR_ADAPTIVE = 3

#: How many questions to select per adaptive round. next_questions is the
#: most expensive interactive call in the engine — about three seconds — so
#: a batch amortises it over several answers rather than paying it per
#: question.
ADAPTIVE_BATCH = 16


@dataclass
class Question:
    item_id: int
    row: dict


@dataclass
class ElicitationSession:
    """Chooses what to ask next, and remembers what has been answered.

    Seeded questions come first, because the adaptive criterion needs a
    posterior to reduce the variance of and there is none yet. Once three
    verdicts exist the session refits and switches to variance reduction.
    """

    fs: object
    meta: dict[int, dict]
    pool: np.ndarray
    asked: set[int]
    languages: tuple[str, ...] = ()
    target: int = 40

    answered: list[tuple[int, float]] = field(default_factory=list)
    _batch: list[int] = field(default_factory=list)
    _cursor: int = 0

    @property
    def answered_count(self) -> int:
        return len(self.answered)

    @property
    def finished(self) -> bool:
        return self.answered_count >= self.target

    def _seed(self, k: int) -> list[int]:
        return [
            item
            for item in elicit.seed_questions(
                self.fs, self.meta, k=k, languages=self.languages, pool=self.pool
            )
            if item not in self.asked
        ]

    def _refill(self) -> None:
        if self.answered_count >= MIN_FOR_ADAPTIVE:
            ids = np.array([a[0] for a in self.answered])
            rewards = np.array([a[1] for a in self.answered])
            model = fit_taste(
                self.fs.vectors_for(ids), rewards, allow_rff=False
            )
            self._batch = elicit.next_questions(
                model, self.fs, self.meta, self.asked,
                k=ADAPTIVE_BATCH, languages=self.languages, pool=self.pool,
            )
        else:
            self._batch = self._seed(self.target * 3)
        self._cursor = 0

    def next_question(self) -> Question | None:
        """The next title to ask about, or None when the pool is exhausted.

        Marks the question as asked before returning it. A question offered
        and then abandoned must not come back — from the pool's point of view
        it has been spent either way.
        """
        while True:
            if self._cursor >= len(self._batch):
                self._refill()
                if not self._batch:
                    return None
            item = self._batch[self._cursor]
            self._cursor += 1
            if item in self.asked:
                continue
            self.asked.add(item)
            return Question(item_id=item, row=self.meta[item])

    def record(self, item_id: int, reward: float) -> None:
        """Note a verdict. The caller owns writing it to the event log."""
        self.answered.append((item_id, reward))

    def start(self) -> None:
        """Prime the first batch from the seeded opening ladder."""
        self._batch = self._seed(max(self.target, 24))
        self._cursor = 0
