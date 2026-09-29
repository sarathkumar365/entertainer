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

**Skill over time** — does the model pull *ahead of its control* as verdicts
accumulate? This is the self-improvement claim, stated as a number, and it is
deliberately not "does raw error fall".

Raw error is not comparable across the log, because how hard the log is to
predict changes as you go. A stretch where every verdict carries the same value
is trivially predictable, so error there is near zero for any model and for the
control alike; a stretch with real spread is harder for both. Trending raw
error therefore measures *how varied the labels happened to be at each point*
and only incidentally the model. On the author's own log this was not a
hypothetical: the first seventeen verdicts all carried the same value, the
model scored a perfect 0.00 over them by predicting a constant, and the
resulting curve read as a model getting significantly worse when what had
actually happened is that it started being tested.

Skill — the control's error minus the model's — is immune to that, because
both face the identical verdict at every step and the difficulty cancels.

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

#: A step whose control error is below this discriminates nothing: predicting
#: the running average was already right, so no model can demonstrate skill
#: there and none can be penalised either. Such steps are excluded from the
#: trend rather than counted as zero, so a long constant opening cannot dilute
#: the slope. On the 0-1 reward scale this is half a tenth of a point out of
#: ten.
UNINFORMATIVE = 0.005


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

    def skill(self) -> list[float]:
        """Per-step advantage over the control: positive means the model won.

        The control faces the same verdict the model does, so the difficulty of
        that particular verdict cancels. That is the entire point — see the
        module docstring.
        """
        return [
            b - m for b, m in zip(self.baseline_error, self.absolute_error, strict=True)
        ]

    def informative(self) -> list[int]:
        """Indices of the steps where skill could be demonstrated at all.

        A step whose control error is essentially zero cannot separate any two
        models, so including it in a trend adds a zero that is evidence of
        nothing while still pulling a regression line.
        """
        return [i for i, b in enumerate(self.baseline_error) if b > UNINFORMATIVE]

    @property
    def n_informative(self) -> int:
        return len(self.informative())

    def trend(self, window: int = 10) -> tuple[float, float]:
        """Mean skill over the first and last window of informative steps.

        Returns ``(early, late)`` in reward units, where *late > early* means
        the model is pulling ahead of the control. Note this is the opposite
        polarity to the error curve this once reported: higher is better.
        """
        idx = self.informative()
        if not idx:
            return float("nan"), float("nan")
        skill = self.skill()
        vals = [skill[i] for i in idx]
        if len(vals) < 2 * window:
            half = max(len(vals) // 2, 1)
            return float(np.mean(vals[:half])), float(np.mean(vals[-half:]))
        return float(np.mean(vals[:window])), float(np.mean(vals[-window:]))

    def learning_slope(self) -> tuple[float, float]:
        """OLS slope of *skill* against step count, with its p-value.

        A significantly **positive** slope is the thing worth reporting: the
        model is pulling further ahead of the control as it sees more of the
        user. Computed over informative steps only.
        """
        idx = self.informative()
        if len(idx) < 8:
            return float("nan"), float("nan")
        skill = self.skill()
        res = stats.linregress([self.steps[i] for i in idx], [skill[i] for i in idx])
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


# Interpretation thresholds. These are judgement calls about what counts as
# evidence, not arithmetic, so they live beside the measurement rather than
# inside whichever renderer happens to display it.
SIGNIFICANCE = 0.05
#: How far observed interval coverage may sit from nominal before the
#: intervals are called miscalibrated. Wide, because coverage on a couple of
#: hundred verdicts is itself a noisy estimate.
CALIBRATION_BAND = 0.12

MIN_VERDICTS = 8


@dataclass
class Reading:
    """One row of an audit: what was measured, and what it means.

    ``tone`` is a word, not markup — "good", "bad", "warn" or "" — so the same
    reading can be printed by rich, returned as JSON, or rendered in a
    browser without any of them knowing about the others.
    """

    measure: str
    value: str
    reading: str
    tone: str = ""


def readings(
    result: PrequentialResult,
    interval: float = 0.90,
    n_verdicts: int | None = None,
) -> list[Reading]:
    """Turn a prequential result into the rows an audit reports.

    The thresholds applied here are the whole point: a slope is only worth
    calling a trend if it is significant *and* positive, and coverage is only
    miscalibrated once it strays beyond a band wide enough to survive its own
    sampling noise.

    Note the polarity. Both the trend and the slope are measured on *skill* —
    the control's error minus the model's — so larger is better, where the
    error curve these once reported wanted smaller. Raw error is not comparable
    across a log whose difficulty varies; see the module docstring for the
    measurement this got wrong.

    Both the slope and the rank correlation are NaN below the minimum sample.
    They must say so rather than rendering "+nan"; that was a real bug.

    ``n_verdicts`` is the size of the history walked, which is *not*
    ``result.n``: the first ``min_train`` verdicts train the model and are
    never predicted, so ``result.n`` is smaller. Reporting the latter under
    the label "verdicts used" understates the history — ten verdicts in
    produced five predictions, and the row said five.
    """
    early, late = result.trend()
    slope, p_slope = result.learning_slope()
    rho, p_rho = result.spearman()
    coverage = result.coverage()

    out = [
        Reading("verdicts used", f"{result.n if n_verdicts is None else n_verdicts}", ""),
        Reading("mean absolute error", f"{result.mae() * 10:.2f} / 10", "lower is better"),
        Reading(
            "vs running-average baseline",
            f"{result.baseline_mae() * 10:.2f} / 10",
            "model wins" if result.mae() < result.baseline_mae() else "model loses",
            "good" if result.mae() < result.baseline_mae() else "bad",
        ),
    ]

    # Steps where predicting the running average was already exactly right
    # cannot separate any two models. They are excluded from the trend, and
    # said so out loud when there are enough of them to change the reading.
    #
    # The note names the rows it applies to. It sits directly beneath the two
    # error rows, which are means over *every* step, and without saying so it
    # reads as though it qualifies them — which would understate the gap
    # between model and control, since a step with nothing to beat contributes
    # near-zero error to both.
    dropped = result.n - result.n_informative
    if dropped:
        out.append(
            Reading(
                "steps that could discriminate",
                f"{result.n_informative} of {result.n}",
                f"{dropped} had nothing to beat; the errors above still count them",
                "dim",
            )
        )

    if early != early or late != late:
        out.append(Reading("skill: first vs last", "\u2014", "need more data"))
    else:
        out.append(
            Reading(
                "skill: first vs last",
                f"{early * 10:+.2f} \u2192 {late * 10:+.2f}",
                "improving" if late > early else "flat or worse",
                "good" if late > early else "warn",
            )
        )

    if slope != slope or p_slope != p_slope:
        out.append(Reading("learning slope", "\u2014", "need more data"))
    else:
        improving = p_slope < SIGNIFICANCE and slope > 0
        out.append(
            Reading(
                "learning slope",
                f"{slope * 100:+.3f} skill per 100 verdicts",
                "significant" if improving else f"p={p_slope:.3f}",
                "good" if improving else "",
            )
        )
        # The unit is carried in the value string above because the terminal
        # table has no room for a second line. A caller with more space can
        # split on the space; see web/routers/insight.

    calibrated = abs(coverage - interval) < CALIBRATION_BAND
    out.append(
        Reading(
            f"{interval:.0%} interval coverage",
            f"{coverage:.2f}",
            "calibrated" if calibrated else "miscalibrated",
            "good" if calibrated else "warn",
        )
    )

    if rho != rho or p_rho != p_rho:
        out.append(Reading("rank correlation", "\u2014", "need more data"))
    else:
        out.append(Reading("rank correlation", f"{rho:+.3f}", f"p={p_rho:.4f}"))

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
