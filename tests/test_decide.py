"""The decision rule's table in docs/agent/DESIGN.md, phase 4, one row per test."""

from __future__ import annotations

import pytest

pytest.importorskip("guessit")

from entertainer.acquire import Plan  # noqa: E402
from entertainer.acquire.core import GB  # noqa: E402
from entertainer.acquire.match import Candidate, Target, score  # noqa: E402
from entertainer.agent.decide import decide  # noqa: E402
from entertainer.errors import NotAcquired  # noqa: E402

VIKRAM = Target("Vikram", 2022, 174, language="Tamil")


def _plan(*labels: str, sizes=None, target=VIKRAM) -> Plan:
    sizes = sizes or [GB] * len(labels)
    cands = [score(target, Candidate("t", lbl, size, None))
             for lbl, size in zip(labels, sizes, strict=True)]
    return Plan(target, sorted(cands, key=lambda c: -c.score))


def test_a_clear_match_is_taken_without_asking():
    d = decide(_plan("Vikram 2022 TAMIL 1080p x265 HEVC"), max_gb=3)
    assert not d.paused and d.picks[0].score >= 13


def test_copies_of_the_same_film_at_other_qualities_are_not_a_conflict():
    d = decide(_plan("Vikram 2022 TAMIL 1080p x265 HEVC", "Vikram 2022 TAMIL 720p",
                     "Vikram (2022) Tamil 2160p"), max_gb=3)
    assert not d.paused and len(d.picks) == 3


def test_a_label_without_the_year_is_asked_about():
    p = _plan("Vikram TAMIL 1080p x265")
    d = decide(p, max_gb=3)
    assert p.usable, "the language name must not be read as part of the title"
    assert d.paused and "year" in d.reason


def test_a_label_without_the_language_is_asked_about():
    d = decide(_plan("Vikram 2022 1080p x265 HEVC"), max_gb=3)
    assert d.paused and "not marked Tamil" in d.reason


def test_a_low_score_is_asked_about():
    d = decide(_plan("Vikram 2022 TAMIL CAM 1080p"), max_gb=3)
    assert d.paused and "under 13" in d.reason


def test_close_copies_in_different_languages_are_asked_about():
    # A 1080p Hindi dub and a 720p Tamil original land within a point of each other.
    d = decide(_plan("Vikram 2022 TAMIL 720p x265 HEVC", "Vikram 2022 Hindi 1080p x265 HEVC"),
               max_gb=3)
    assert d.paused and "language" in d.reason and len(d.offer) == 2


def test_a_dub_well_below_the_original_is_not_a_conflict_and_is_never_the_fallback():
    d = decide(_plan("Vikram 2022 TAMIL 1080p x265 HEVC", "Vikram 2022 Hindi 480p"), max_gb=3)
    assert not d.paused
    assert [c.lang for c in d.picks] == ["tamil"]


def test_a_multi_audio_copy_carrying_the_original_agrees_with_it():
    d = decide(_plan("Vikram 2022 TAMIL 1080p x265", "Vikram 2022 Tamil + Hindi 1080p x265"),
               max_gb=3)
    assert not d.paused and len(d.picks) == 2


def test_close_copies_of_different_years_are_asked_about():
    # The year guard allows a year either side; two such copies are two claims.
    target = Target("Nosferatu", 1922, 94)
    d = decide(_plan("Nosferatu 1922 1080p x265", "Nosferatu 1923 1080p x265", target=target),
               max_gb=3)
    assert d.paused and "year" in d.reason


def test_close_copies_with_different_titles_are_asked_about():
    target = Target("Vikram", 2022, 174)
    d = decide(_plan("Vikram 2022 1080p x265", "Vikramm 2022 1080p x265", target=target),
               max_gb=3)
    assert d.paused and "title" in d.reason


def test_the_original_and_english_titles_are_one_film():
    target = Target("Spirited Away", 2001, 125, alt_titles=("Sen to Chihiro",))
    p = _plan("Spirited Away 2001 1080p x265", "Sen to Chihiro 2001 1080p x265", target=target)
    d = decide(p, max_gb=3)
    assert {c.title for c in p.usable} == {"spirited away"}
    assert not d.paused


def test_when_every_copy_is_over_the_cap_the_smallest_is_offered():
    d = decide(_plan("Vikram 2022 TAMIL 2160p", "Vikram 2022 TAMIL 1080p x265",
                     sizes=[9 * GB, 4 * GB]), max_gb=3)
    assert d.paused and d.over_cap and [c.size for c in d.offer] == [4 * GB]


def test_an_over_cap_copy_is_passed_over_for_one_that_fits():
    d = decide(_plan("Vikram 2022 TAMIL 2160p HEVC", "Vikram 2022 TAMIL 1080p x265 HEVC",
                     sizes=[9 * GB, 2 * GB]), max_gb=3)
    assert not d.paused and all(c.size <= 3 * GB for c in d.picks)


def test_a_pause_offers_at_most_three():
    d = decide(_plan(*(f"Vikram 2022 {r}" for r in ("1080p", "720p", "2160p", "576p"))),
               max_gb=3)
    assert d.paused and len(d.offer) == 3


def test_nothing_usable_is_a_refusal_not_a_pause():
    with pytest.raises(NotAcquired, match="No source had"):
        decide(_plan("Vikram.1986.720p.mkv"), max_gb=3)
