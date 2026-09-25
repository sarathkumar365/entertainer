"""The audit's interpretation thresholds.

Lifted out of `ent audit`'s render loop, where "is this significant",
"are the intervals calibrated" and "is the model beating the baseline" were
decided inline between two Table.add_row calls. /api/audit needs the same
judgements without the markup.
"""

from __future__ import annotations

import math

from entertainer.evaluation.prequential import (
    CALIBRATION_BAND,
    SIGNIFICANCE,
    PrequentialResult,
    readings,
)


def result(errors, baseline, predicted=None, actual=None, inside=None) -> PrequentialResult:
    res = PrequentialResult()
    res.steps = list(range(5, 5 + len(errors)))
    res.absolute_error = list(errors)
    res.baseline_error = list(baseline)
    res.predicted = list(predicted if predicted is not None else errors)
    res.actual = list(actual if actual is not None else errors)
    res.inside_interval = list(inside if inside is not None else [True] * len(errors))
    return res


def row(rows, measure):
    return next(r for r in rows if r.measure == measure)


def test_verdicts_used_is_the_history_not_the_prediction_count():
    """The first min_train verdicts train the model and are never predicted,
    so result.n is smaller than the history. Reporting it under "verdicts
    used" understated the history — ten in, and the row said five."""
    res = result([0.1] * 5, [0.2] * 5)
    assert row(readings(res, n_verdicts=10), "verdicts used").value == "10"
    assert row(readings(res), "verdicts used").value == "5"


def test_beating_the_baseline_is_called_out_as_a_win():
    rows = readings(result([0.1] * 5, [0.3] * 5))
    assert row(rows, "vs running-average baseline").reading == "model wins"
    assert row(rows, "vs running-average baseline").tone == "good"


def test_losing_to_the_baseline_is_not_softened():
    rows = readings(result([0.4] * 5, [0.2] * 5))
    assert row(rows, "vs running-average baseline").reading == "model loses"
    assert row(rows, "vs running-average baseline").tone == "bad"


def test_a_nan_slope_says_so_rather_than_printing_nan():
    """Below the minimum sample both the slope and the rank correlation are
    NaN. Rendering them unguarded produced "+nan per 100 verdicts"."""
    res = result([0.2] * 4, [0.2] * 4)
    assert math.isnan(res.learning_slope()[0])
    slope = row(readings(res), "learning slope")
    assert slope.value == "—"
    assert slope.reading == "need more data"
    assert row(readings(res), "rank correlation").value == "—"


def test_an_improving_slope_counts_only_when_significant_and_positive():
    """Polarity. The slope is measured on skill — the control's error minus
    the model's — so pulling *ahead* of the control is a rising line. It used
    to be measured on raw error, where improving meant falling."""
    gaining = result([1.0 - i * 0.1 for i in range(12)], [0.5] * 12)
    slope, p = gaining.learning_slope()
    assert slope > 0 and p < SIGNIFICANCE
    assert row(readings(gaining), "learning slope").reading == "significant"

    flat = result([0.5, 0.52, 0.48, 0.51, 0.49, 0.5, 0.51, 0.49, 0.5, 0.5], [0.5] * 10)
    assert row(readings(flat), "learning slope").reading.startswith("p=")


def test_a_model_losing_ground_is_not_called_improving():
    losing = result([0.1 + i * 0.05 for i in range(12)], [0.5] * 12)
    slope, _ = losing.learning_slope()
    assert slope < 0
    assert row(readings(losing), "learning slope").reading != "significant"
    assert row(readings(losing), "skill: first vs last").reading == "flat or worse"


def test_steps_with_nothing_to_beat_are_reported_and_excluded():
    """The author's log opened with seventeen identical verdicts, where the
    running average was already exactly right. Counting those as ties let a
    constant opening drag the trend; they are dropped, and the drop is said
    out loud rather than silently changing the denominator."""
    res = result([0.0] * 10 + [0.1] * 10, [0.0] * 10 + [0.3] * 10)
    rows = readings(res)
    discriminating = row(rows, "steps that could discriminate")
    assert discriminating.value == "10 of 20"
    assert discriminating.reading == "10 had nothing to beat"
    assert discriminating.tone == "dim"

    # And the surviving ten all show the same +0.20 gain, so the trend is flat
    # rather than the sharp rise that including the zeros would manufacture.
    early, late = res.trend(window=5)
    assert early == late


def test_no_discriminating_row_when_every_step_counted():
    rows = readings(result([0.1] * 10, [0.3] * 10))
    assert not any(r.measure == "steps that could discriminate" for r in rows)


def test_skill_is_rendered_with_its_sign():
    """A bare "2.00 → 0.50" reads as an error falling. The sign is what says
    these are gains over the control, in points out of ten."""
    rows = readings(result([0.1] * 10, [0.3] * 10))
    assert row(rows, "skill: first vs last").value == "+2.00 \u2192 +2.00"


def test_coverage_within_the_band_is_calibrated():
    inside = [True] * 9 + [False]
    rows = readings(result([0.1] * 10, [0.2] * 10, inside=inside), interval=0.90)
    assert row(rows, "90% interval coverage").reading == "calibrated"


def test_coverage_outside_the_band_is_flagged():
    inside = [True] * 5 + [False] * 5
    rows = readings(result([0.1] * 10, [0.2] * 10, inside=inside), interval=0.90)
    assert abs(0.5 - 0.90) > CALIBRATION_BAND
    assert row(rows, "90% interval coverage").reading == "miscalibrated"
    assert row(rows, "90% interval coverage").tone == "warn"


def test_tone_is_a_word_not_markup():
    """Library code must not emit rich tags; the renderer maps tone to style."""
    for r in readings(result([0.1] * 10, [0.2] * 10)):
        assert "[" not in r.reading and "[" not in r.value
        assert r.tone in ("", "good", "bad", "warn", "dim")


def test_a_model_reports_real_verdicts_not_sampled_negatives():
    """`ent taste` said "learned from 1166 verdicts" against 169 real ones —
    n_obs counts the 1000 sampled pseudo-negatives the fit is padded with."""
    import numpy as np

    from entertainer.models.taste import fit

    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6))
    y = rng.uniform(size=40)
    model = fit(X, y, allow_rff=False, capacity_obs=12)
    assert model.n_obs == 40
    assert model.n_real == 12


def test_n_real_survives_a_save_and_load(tmp_path, monkeypatch):
    import numpy as np

    from entertainer.models.taste import TasteModel, fit

    rng = np.random.default_rng(0)
    model = fit(rng.normal(size=(30, 4)), rng.uniform(size=30), allow_rff=False, capacity_obs=7)
    path = tmp_path / "taste.npz"
    model.to_npz(path)
    assert TasteModel.from_npz(path).n_real == 7


def test_a_model_saved_before_n_real_existed_still_loads(tmp_path):
    """Falling back to n_obs is what those files were displaying anyway."""
    import json

    import numpy as np

    from entertainer.models.taste import TasteModel, fit

    rng = np.random.default_rng(0)
    model = fit(rng.normal(size=(20, 3)), rng.uniform(size=20), allow_rff=False)
    path = tmp_path / "old.npz"
    fm = model.feature_map
    np.savez_compressed(
        path,
        mean=model.mean, cov=model.cov,
        alpha=np.array([model.alpha]), beta=np.array([model.beta]),
        n_obs=np.array([model.n_obs]), y_mean=np.array([model.y_mean]),
        log_evidence=np.array([model.log_evidence]),
        spec=np.array([json.dumps(fm.to_dict())]),
        W=fm.W if fm.W is not None else np.zeros(0, dtype=np.float32),
        b=fm.b if fm.b is not None else np.zeros(0, dtype=np.float32),
    )
    loaded = TasteModel.from_npz(path)
    assert loaded.n_real == loaded.n_obs == model.n_obs
