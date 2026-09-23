"""What to run next, and when coverage is worth complaining about.

Both rules lived in cli.py. next_step returned a single string with rich
markup baked into it, which made it unusable anywhere that is not a terminal
— the planned /api/mode endpoint would have had to reimplement it.
"""

from __future__ import annotations

from entertainer import pipeline

FULL = {
    "catalog": True,
    "content_embeddings": True,
    "cf_factors": True,
    "fused_space": True,
    "taste_model": True,
}


def missing(**overrides):
    return {**FULL, **overrides}


def test_first_gap_wins_in_build_order():
    """Dependency order is load-bearing: fuse needs cf, cf needs embeddings."""
    assert pipeline.next_step(missing(catalog=False), 0).command == "ent setup"
    assert pipeline.next_step(missing(content_embeddings=False), 0).command == "ent data embed"
    assert pipeline.next_step(missing(cf_factors=False), 0).command == "ent data cf"
    assert pipeline.next_step(missing(fused_space=False), 0).command == "ent data fuse"


def test_an_earlier_gap_hides_a_later_one():
    step = pipeline.next_step(missing(content_embeddings=False, fused_space=False), 0)
    assert step.command == "ent data embed"


def test_a_built_pipeline_with_no_verdicts_asks_for_verdicts():
    step = pipeline.next_step(FULL, 0)
    assert step.command == "ent onboard"
    assert "bulk" in step.note


def test_between_the_two_thresholds_it_recommends_and_mentions_the_audit():
    step = pipeline.next_step(FULL, pipeline.MIN_FOR_RECS)
    assert step.command == "ent recs"
    assert "audit" in step.note


def test_past_the_audit_threshold_there_is_no_aside():
    assert pipeline.next_step(FULL, pipeline.MIN_FOR_AUDIT) == pipeline.Step("ent recs")


def test_the_command_carries_no_markup():
    """It used to, which is why this could only ever be printed by rich."""
    for n in (0, 1, 3, 8, 500):
        step = pipeline.next_step(FULL, n)
        assert "[" not in step.command and "[" not in step.note


def test_coverage_is_only_reported_when_it_is_actually_short():
    assert pipeline.coverage_is_stale({"catalog": 1000, "fused": 1000}) is None
    assert pipeline.coverage_is_stale({"catalog": 1000, "fused": 996}) is None


def test_a_shortfall_returns_the_share_so_the_caller_need_not_recompute_it():
    share = pipeline.coverage_is_stale({"catalog": 1000, "fused": 900})
    assert share == 0.9


def test_an_empty_catalogue_is_not_a_coverage_problem():
    """Zero over zero, and nothing to warn about anyway."""
    assert pipeline.coverage_is_stale({"catalog": 0, "fused": 0}) is None
