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


def test_an_improving_slope_counts_only_when_significant_and_negative():
    falling = result([1.0 - i * 0.1 for i in range(12)], [0.5] * 12)
    slope, p = falling.learning_slope()
    assert slope < 0 and p < SIGNIFICANCE
    assert row(readings(falling), "learning slope").reading == "significant"

    flat = result([0.5, 0.52, 0.48, 0.51, 0.49, 0.5, 0.51, 0.49, 0.5, 0.5], [0.5] * 10)
    assert row(readings(flat), "learning slope").reading.startswith("p=")


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
