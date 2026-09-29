"""The knobs a full build turns.

`ent setup` has no CLI-level coverage — it downloads tens of gigabytes and
runs for about three hours — so its tuning constants were literals inside a
command body that nothing ever read. Several of them have to agree with each
other, and nothing said so.
"""

from __future__ import annotations

import pytest

from entertainer.pipeline import STAGES, BuildSettings


def test_the_defaults_are_the_ones_the_real_build_used():
    """Pinned so a change is deliberate. These produced the current
    73,494-title catalogue and its 192-dimensional item space."""
    s = BuildSettings()
    assert (s.min_votes, s.floor_scale) == (50, 1.0)
    assert (s.concurrency, s.keyword_target) == (40, 150_000)
    assert (s.cf_factors, s.cf_iterations, s.cf_holdout) == (192, 20, 2_000)
    assert s.cf_signal == "watched"
    assert s.fusion_dim == 192


def test_the_fused_space_cannot_be_wider_than_the_factorisation():
    """The fused space is a rotation of the collaborative and content blocks.
    Asking for more dimensions than there are factors pads it with noise, and
    nothing downstream would notice."""
    with pytest.raises(ValueError, match="would be noise"):
        BuildSettings(fusion_dim=256, cf_factors=192)


def test_a_narrower_fused_space_is_allowed():
    assert BuildSettings(fusion_dim=64).fusion_dim == 64


def test_every_stage_has_a_label_and_they_are_in_dependency_order():
    names = [name for name, _ in STAGES]
    assert names == [
        "sources", "catalogue", "tmdb", "prune", "embeddings", "cf", "fusion",
    ]
    assert all(label for _, label in STAGES)
    # prune before embeddings, so nothing is encoded that is about to be
    # deleted; embeddings and cf before fusion, which combines both towers.
    assert names.index("prune") < names.index("embeddings")
    assert names.index("embeddings") < names.index("fusion")
    assert names.index("cf") < names.index("fusion")
    # Fusion is the last stage: the population prior that used to follow it was
    # cut on 25 September 2026, measured at or below the arm that omitted it.
    assert names[-1] == "fusion"


def test_the_reporter_stages_match_the_declared_ones():
    from entertainer.build_events import STAGES as REPORTED

    declared = [name for name, _ in STAGES]
    reported = [s if isinstance(s, str) else s[0] for s in REPORTED]
    assert set(declared) == set(reported), (
        "pipeline.STAGES and build_events.STAGES describe the same build; "
        "the Build Studio reads the latter"
    )
