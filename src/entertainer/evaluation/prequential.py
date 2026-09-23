"""Is the engine actually getting better at *you*?

The simulation in ``simulate.py`` proves the method works on strangers. That
is necessary but it is not the claim the user cares about. The claim they care
about is that the thing improves as they feed it, and the only honest way to
check that is on their own history.

This runs a prequential (progressive validation) evaluation: walk the verdict
log in chronological order, and at each step fit the model on everything
before that point and predict what comes next. Every prediction is made by a
model that has never seen the answer, so the resulting error curve is an
honest out-of-sample learning curve rather than a fit statistic.

Three things are reported.

**Accuracy over time** — does prediction error fall as verdicts accumulate?
This is the self-improvement claim, stated as a number.

**Calibration** — when the model says "8.2 ± 0.4", is it right about the
±? A model whose error bars are wrong is worse than one with no error bars,
because the exploration policy is built on them. Measured as the fraction of
outcomes inside the 90% predictive interval, which should sit near 0.90.

**Ranking** — Spearman correlation between predicted and actual preference
over the held-out tail, which is what recommendation quality actually rests
on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from ..models.features import FeatureSpace
from ..models.taste import fit as fit_taste

# Below this the posterior is still essentially the prior and the numbers are
# noise rather than evidence.
MIN_TRAIN = 5


@dataclass
class PrequentialResult:
    steps: list[int] = field(default_factory=list)
    absolute_error: list[float] = field(default_factory=list)
    predicted: list[float] = field(default_factory=list)
    actual: list[float] = field(default_factory=list)
    inside_interval: list[bool] = field(default_factory=list)
    baseline_error: list[float] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.steps)

    def mae(self) -> float:
        return float(np.mean(self.absolute_error)) if self.absolute_error else float("nan")

    def baseline_mae(self) -> float:
        return float(np.mean(self.baseline_error)) if self.baseline_error else float("nan")

    def coverage(self) -> float:
        return float(np.mean(self.inside_interval)) if self.inside_interval else float("nan")

    def spearman(self) -> tuple[float, float]:
        if self.n < 8:
            return float("nan"), float("nan")
        r = stats.spearmanr(self.predicted, self.actual)
        return float(r.statistic), float(r.pvalue)

    def trend(self, window: int = 10) -> tuple[float, float]:
        """Mean error over the first and last window of predictions."""
        if self.n < 2 * window:
            half = max(self.n // 2, 1)
            early = float(np.mean(self.absolute_error[:half]))
            late = float(np.mean(self.absolute_error[-half:]))
            return early, late
        return (
            float(np.mean(self.absolute_error[:window])),
            float(np.mean(self.absolute_error[-window:])),
        )

    def learning_slope(self) -> tuple[float, float]:
        """OLS slope of error against step count, with its p-value.

        A significantly negative slope is the thing worth reporting: the model
        is measurably improving as it sees more of the user.
        """
        if self.n < 8:
            return float("nan"), float("nan")
        res = stats.linregress(self.steps, self.absolute_error)
        return float(res.slope), float(res.pvalue)


def run(
    fs: FeatureSpace,
    ordered_items: list[int],
    ordered_rewards: list[float],
    min_train: int = MIN_TRAIN,
    interval: float = 0.90,
) -> PrequentialResult:
    """Walk the verdict history forwards, predicting each unseen verdict."""
    out = PrequentialResult()
    n = len(ordered_items)
    if n <= min_train:
        return out

    z = float(stats.norm.ppf(0.5 + interval / 2.0))
    X = fs.vectors_for(ordered_items)
    y = np.asarray(ordered_rewards, dtype=np.float64)

    for i in range(min_train, n):
        model = fit_taste(X[:i], y[:i], allow_rff=(i >= 150))
        mean, sd = model.predict(X[i : i + 1])
        err = abs(float(mean[0]) - float(y[i]))

        out.steps.append(i)
        out.absolute_error.append(err)
        out.predicted.append(float(mean[0]))
        out.actual.append(float(y[i]))
        out.inside_interval.append(abs(float(mean[0]) - float(y[i])) <= z * float(sd[0]))
        # The control: predict the running mean of everything seen so far.
        # Any model that cannot beat this has learned nothing about the person.
        out.baseline_error.append(abs(float(y[:i].mean()) - float(y[i])))
    return out


# Below this many logged recommendations with an outcome, the importance
# weights have ruinous variance and the estimate is not worth reporting.
MIN_LOGGED = 30


def snips(
    logged: list[tuple[float, float, float]],
    new_scores: list[float],
    temperature: float = 1.0,
) -> float | None:
    """Self-normalised inverse propensity estimate of a new policy's value.

    ``logged``: (reward, propensity under the logging policy, logged score).

    This answers "what would my *current* model have scored, had it been the
    one choosing back then" using only data the old model collected. It is the
    reason every slate records the probability each item had of being shown.

    Self-normalised rather than vanilla IPS because raw importance weights on
    a few hundred samples have ruinous variance; SNIPS trades a little bias
    for a large reduction in it. With very few logged interactions the
    estimate is still fragile, so it returns None rather than a number that
    would be over-read.
    """
    if len(logged) < MIN_LOGGED:
        return None
    rewards = np.array([r for r, _, _ in logged], dtype=np.float64)
    props = np.array([max(p, 1e-6) for _, p, _ in logged], dtype=np.float64)
    scores = np.array(new_scores, dtype=np.float64)

    logits = (scores - scores.max()) / max(temperature, 1e-6)
    new_probs = np.exp(logits)
    new_probs /= new_probs.sum()

    weights = new_probs / props
    # Clip to the 99th percentile: a single enormous weight otherwise decides
    # the whole estimate.
    weights = np.minimum(weights, np.quantile(weights, 0.99))
    denom = weights.sum()
    return float((weights * rewards).sum() / denom) if denom > 0 else None
