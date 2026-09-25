import numpy as np
import pytest

from entertainer.evaluation.prequential import (
    SIGNIFICANCE,
    PrequentialResult,
    run,
    snips,
)
from entertainer.models.features import FeatureSpace


def _space(n=400, d=24, seed=0):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, d)).astype(np.float32)
    latent /= np.linalg.norm(latent, axis=1, keepdims=True)
    side = np.zeros((n, 5), dtype=np.float32)
    ids = np.arange(n, dtype=np.int32)
    return FeatureSpace(ids, latent, side, {int(i): i for i in range(n)}), rng


def test_skill_grows_as_verdicts_accumulate():
    fs, rng = _space()
    w = rng.normal(size=fs.n_latent)
    w /= np.linalg.norm(w)
    rewards = np.clip(0.5 + 0.4 * (fs.latent @ w) + rng.normal(0, 0.03, len(fs.item_ids)), 0, 1)

    res = run(fs, list(range(300)), list(rewards[:300]))
    early, late = res.trend(window=30)
    assert late > early, "skill is a gain over the control, so larger is better"
    assert res.mae() < res.baseline_mae(), "must beat predicting the running average"
    slope, p = res.learning_slope()
    assert slope > 0 and p < 0.05


def test_a_constant_opening_does_not_read_as_decline():
    """The bug this measurement was rebuilt for.

    The author's log opened with seventeen identical verdicts. A model fitted
    on a constant predicts that constant and scores a perfect 0.00 over them —
    not accuracy, an absence of variance to be wrong about. Trending *raw
    error* then compared that stretch against a later one with real spread and
    reported a significant decline, when what had actually happened is that the
    model started being tested.

    Skill is immune: over the constant opening the control is exactly as right
    as the model, so neither gains, and the steps are dropped as
    uninformative rather than counted as ties.
    """
    fs, rng = _space()
    w = rng.normal(size=fs.n_latent)
    w /= np.linalg.norm(w)
    real = np.clip(0.5 + 0.4 * (fs.latent @ w) + rng.normal(0, 0.03, len(fs.item_ids)), 0, 1)

    items = list(range(220))
    rewards = [0.75] * 20 + list(real[20:220])
    res = run(fs, items, rewards)

    # The constant opening is exactly the stretch that discriminates nothing.
    assert res.n_informative < res.n
    opening = [s for s in res.skill()[:10]]
    assert max(abs(s) for s in opening) < 1e-6, "no model can gain on a constant"

    slope, p = res.learning_slope()
    assert not (slope < 0 and p < SIGNIFICANCE), (
        "a constant opening must not manufacture a significant decline"
    )


def test_uninformative_steps_are_excluded_from_the_trend():
    """Not merely counted as zero — a long enough run of them would otherwise
    drag a regression line without being evidence of anything."""
    res = PrequentialResult()
    res.steps = list(range(5, 45))
    # First twenty: the control was already exactly right. Last twenty: the
    # model wins by a steady margin.
    res.baseline_error = [0.0] * 20 + [0.30] * 20
    res.absolute_error = [0.0] * 20 + [0.10] * 20
    res.predicted = res.actual = [0.5] * 40
    res.inside_interval = [True] * 40

    assert res.n_informative == 20
    early, late = res.trend(window=5)
    assert early == late == pytest.approx(0.20), (
        "the trend must see only the twenty steps that could discriminate"
    )


def test_intervals_are_calibrated_on_a_well_specified_problem():
    fs, rng = _space(n=300)
    w = rng.normal(size=fs.n_latent)
    w /= np.linalg.norm(w)
    rewards = np.clip(0.5 + 0.3 * (fs.latent @ w) + rng.normal(0, 0.05, 300), 0, 1)
    res = run(fs, list(range(250)), list(rewards[:250]))
    # A correct 90% interval should contain roughly 90% of outcomes.
    assert 0.78 <= res.coverage() <= 1.0


def test_pure_noise_does_not_produce_a_learning_claim():
    """Guards against the failure mode where the harness flatters itself."""
    fs, rng = _space(n=300)
    rewards = rng.uniform(size=300)
    res = run(fs, list(range(200)), list(rewards[:200]))
    slope, p = res.learning_slope()
    assert not (slope > 0 and p < 0.01), "should not claim learning from noise"


def test_too_little_history_returns_nothing():
    fs, _ = _space(n=20)
    assert run(fs, list(range(4)), [0.5] * 4).n == 0


def test_snips_requires_enough_logged_data():
    assert snips([(1.0, 0.1, 0.0)] * 10, [0.0] * 10) is None


def test_snips_prefers_a_policy_that_upweights_rewarding_items():
    rng = np.random.default_rng(0)
    n = 200
    rewards = rng.uniform(size=n)
    props = np.full(n, 1.0 / n)
    logged = [(float(r), float(p), 0.0) for r, p in zip(rewards, props, strict=True)]

    good = snips(logged, list(rewards * 5.0))     # scores aligned with reward
    bad = snips(logged, list(-rewards * 5.0))     # scores anti-aligned
    assert good is not None and bad is not None
    assert good > bad
