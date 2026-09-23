"""Netflix import: parsing, grading, and the guards against wrong attachment.

The whole point of the grading is that a wrong match is worse than a missing
one, so most of these tests pin refusals rather than successes.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from entertainer.data import netflix


def _hit(tmdb_id, title, year, kind="movie", popularity=1.0, votes=0, original=None):
    key = "release_date" if kind == "movie" else "first_air_date"
    name = "title" if kind == "movie" else "name"
    orig = "original_title" if kind == "movie" else "original_name"
    return {
        "id": tmdb_id, "_kind": kind, name: title, orig: original or title,
        key: f"{year}-01-01", "popularity": popularity, "vote_count": votes,
        "original_language": "en",
    }


@pytest.fixture
def fake_search(monkeypatch):
    calls: dict[str, list[dict]] = {}

    def _search(query, year=None, kind=None):
        return calls.get(query, [])

    monkeypatch.setattr(netflix.tmdb, "search", _search)
    return calls


# --- parsing ----------------------------------------------------------------


def test_parses_the_real_jsongraph_shape():
    blob = {
        "jsonGraph": {"aui": {"ratingHistory": {"$type": "atom", "value": {
            "page": 0, "totalRatings": 2, "ratingItems": [
                {"title": "Dangal", "reactionRatings": {"thumbs": "THUMBS_UP"},
                 "movieID": 80166185, "date": "26-2-14"},
                {"title": "Sahara", "reactionRatings": {"thumbs": "THUMBS_DOWN"},
                 "movieID": 70000199, "date": "25-12-05"},
            ]}}}}
    }
    ratings = netflix.parse(blob)
    assert [r.title for r in ratings] == ["Dangal", "Sahara"]
    assert ratings[0].verdict == "like"
    assert ratings[1].verdict == "dislike"
    assert ratings[0].rated_on == date(2026, 2, 14)


def test_parses_a_bare_rating_items_object():
    assert len(netflix.parse({"ratingItems": [
        {"title": "X", "thumbs": "THUMBS_UP", "movieID": 1, "date": "25-1-01"}
    ]})) == 1


def test_unknown_thumbs_values_are_dropped_not_guessed():
    ratings = netflix.parse({"ratingItems": [
        {"title": "X", "thumbs": "THUMBS_SIDEWAYS", "movieID": 1, "date": "25-1-01"},
        {"title": "Y", "thumbs": "THUMBS_UP", "movieID": 2, "date": "25-1-01"},
    ]})
    assert [r.title for r in ratings] == ["Y"]


def test_every_thumbs_value_names_a_verdict_the_model_knows():
    # The reward lives in one place. If these names ever stop resolving, the
    # import would either crash or, worse, record a number nothing else uses.
    from entertainer.models.taste import verdict_to_reward

    assert netflix.THUMB_VERDICTS["THUMBS_WAY_UP"] == "love"
    assert netflix.THUMB_VERDICTS["THUMBS_UP"] == "like"
    assert netflix.THUMB_VERDICTS["THUMBS_DOWN"] == "dislike"
    rewards = [verdict_to_reward(v) for v in netflix.THUMB_VERDICTS.values()]
    assert rewards == sorted(rewards, reverse=True)


@pytest.mark.parametrize(
    "raw,expected",
    [("26-9-18", date(2026, 9, 18)), ("25-12-05", date(2025, 12, 5)),
     ("", None), ("nonsense", None), ("25-13-01", None)],
)
def test_two_digit_netflix_dates(raw, expected):
    assert netflix.parse_date(raw) == expected


def test_title_normalisation_ignores_case_accents_and_punctuation():
    assert netflix.normalise("The Pope's Exorcist") == netflix.normalise("the popes exorcist")
    assert netflix.normalise("Ekō") == netflix.normalise("Eko")
    assert netflix.normalise("Rebel Moon — Part One") == "rebel moon part one"


# --- grading ----------------------------------------------------------------


def test_a_unique_exact_match_resolves_automatically(fake_search):
    fake_search["Dangal"] = [_hit(1, "Dangal", 2016, votes=1221)]
    res = netflix.resolve_one(netflix.Rating("Dangal", "THUMBS_UP", 1, date(2026, 2, 14)))
    assert res.confidence == "high"
    assert res.automatic
    assert res.match["tmdb_id"] == 1


def test_two_films_sharing_a_title_are_not_resolved_automatically(fake_search):
    fake_search["Hunger"] = [
        _hit(1, "Hunger", 2008, popularity=4.7, votes=1257),
        _hit(2, "Hunger", 2023, popularity=3.3, votes=499),
    ]
    res = netflix.resolve_one(netflix.Rating("Hunger", "THUMBS_WAY_UP", 1, date(2026, 8, 15)))
    assert res.confidence == "ambiguous"
    assert not res.automatic
    assert res.alternatives, "the runner-up must be shown so a human can choose"


def test_a_decisive_popularity_gap_does_resolve(fake_search):
    fake_search["Prometheus"] = [
        _hit(1, "Prometheus", 2012, popularity=22.1, votes=13643),
        _hit(2, "Prometheus", 2015, popularity=0.1, votes=0),
    ]
    res = netflix.resolve_one(netflix.Rating("Prometheus", "THUMBS_WAY_UP", 1, date(2026, 7, 11)))
    assert res.confidence == "high"


def test_a_popular_but_less_voted_candidate_does_not_win_on_popularity_alone(fake_search):
    # Recent releases carry inflated TMDB popularity. Without the vote check a
    # brand-new film with 29 votes would outrank the 1989 film actually watched.
    fake_search["Best of the Best"] = [
        _hit(1, "Best of the Best", 2026, popularity=53.4, votes=29),
        _hit(2, "Best of the Best", 1989, popularity=1.0, votes=356),
    ]
    res = netflix.resolve_one(
        netflix.Rating("Best of the Best", "THUMBS_UP", 1, date(2026, 9, 18))
    )
    assert res.confidence == "ambiguous"


def test_a_film_released_after_the_rating_date_is_excluded(fake_search):
    fake_search["Alpha"] = [
        _hit(1, "Alpha", 2030, popularity=90.0, votes=10),
        _hit(2, "Alpha", 2018, popularity=10.1, votes=2823),
    ]
    res = netflix.resolve_one(netflix.Rating("Alpha", "THUMBS_WAY_UP", 1, date(2025, 12, 13)))
    assert res.confidence == "high"
    assert res.match["year"] == 2018


def test_no_exact_match_is_graded_low_and_never_written(fake_search):
    fake_search["Blast"] = [_hit(1, "Blast Furnace", 2020, popularity=50.0, votes=900)]
    res = netflix.resolve_one(netflix.Rating("Blast", "THUMBS_UP", 1, date(2026, 7, 10)))
    assert res.confidence == "low"
    assert not res.automatic


def test_an_empty_title_is_unresolvable(fake_search):
    res = netflix.resolve_one(netflix.Rating(" ".strip(), "THUMBS_UP", 70136118, date(2025, 7, 20)))
    assert res.confidence == "unresolvable"
    assert res.match is None


def test_edition_markers_are_stripped_before_searching(fake_search):
    fake_search["Pushpa 2: The Rule"] = [_hit(1, "Pushpa 2: The Rule", 2024, votes=900)]
    res = netflix.resolve_one(
        netflix.Rating("Pushpa 2: The Rule (Reloaded Version)", "THUMBS_UP", 1, date(2025, 6, 21))
    )
    assert res.confidence == "high"


def test_an_original_language_title_can_carry_the_match(fake_search):
    fake_search["Oru Durooha Saahacharyathil"] = [
        _hit(1, "In a Suspicious Situation", 2024, votes=40,
             original="Oru Durooha Saahacharyathil")
    ]
    res = netflix.resolve_one(
        netflix.Rating("Oru Durooha Saahacharyathil", "THUMBS_UP", 1, date(2026, 5, 18))
    )
    assert res.confidence == "high"


# --- the collision guard ----------------------------------------------------


def test_two_ratings_landing_on_one_film_are_both_demoted(fake_search):
    # The Chinese original and the Korean remake of "A Love So Beautiful" are
    # separate Netflix entries with opposite verdicts. Writing both against one
    # catalogue row would put contradictory labels on the same item.
    fake_search["A Love So Beautiful"] = [_hit(1, "A Love So Beautiful", 2017, kind="tv", votes=87)]
    ratings = [
        netflix.Rating("A Love So Beautiful", "THUMBS_UP", 80239640, date(2026, 6, 21)),
        netflix.Rating("A Love So Beautiful", "THUMBS_DOWN", 81354739, date(2026, 6, 21)),
    ]
    resolved = netflix.flag_collisions(netflix.resolve(ratings))
    assert [r.confidence for r in resolved] == ["ambiguous", "ambiguous"]


def test_the_collision_guard_leaves_distinct_films_alone(fake_search):
    fake_search["Dangal"] = [_hit(1, "Dangal", 2016, votes=1221)]
    fake_search["Sisu"] = [_hit(2, "Sisu", 2022, votes=2978)]
    resolved = netflix.flag_collisions(netflix.resolve([
        netflix.Rating("Dangal", "THUMBS_UP", 1, date(2026, 2, 14)),
        netflix.Rating("Sisu", "THUMBS_UP", 2, date(2025, 6, 24)),
    ]))
    assert [r.confidence for r in resolved] == ["high", "high"]


def test_parse_accepts_a_path(tmp_path, fake_search):
    path = tmp_path / "page.json"
    path.write_text(json.dumps({"ratingItems": [
        {"title": "X", "thumbs": "THUMBS_UP", "movieID": 1, "date": "25-1-01"}
    ]}), encoding="utf-8")
    assert len(netflix.parse(path)) == 1


# --- human rulings ----------------------------------------------------------


def test_accept_promotes_the_best_guess(fake_search):
    fake_search["Extinction"] = [
        _hit(1, "Extinction", 2015, popularity=6.4, votes=835),
        _hit(2, "Extinction", 2018, popularity=5.6, votes=2062),
    ]
    resolved = netflix.resolve([netflix.Rating("Extinction", "THUMBS_UP", 99, date(2026, 8, 12))])
    assert resolved[0].confidence == "ambiguous"
    netflix.apply_decisions(resolved, {"99": "accept"})
    assert resolved[0].confidence == "high"
    assert resolved[0].match["tmdb_id"] == 1


def test_an_explicit_choice_overrides_the_best_guess(fake_search):
    fake_search["Hunger"] = [
        _hit(1, "Hunger", 2008, popularity=4.7, votes=1257),
        _hit(2, "Hunger", 2023, popularity=3.3, votes=499),
    ]
    resolved = netflix.resolve([netflix.Rating("Hunger", "THUMBS_WAY_UP", 81517155, date(2026, 8, 15))])
    netflix.apply_decisions(
        resolved, {"81517155": {"tmdb_id": 1071806, "kind": "movie", "year": 2023}}
    )
    assert resolved[0].confidence == "high"
    assert resolved[0].match["tmdb_id"] == 1071806


def test_skip_is_never_written(fake_search):
    fake_search["Laapataa Ladies"] = [_hit(1, "Lost Ladies", 2024, votes=300)]
    resolved = netflix.resolve([netflix.Rating("Laapataa Ladies", "THUMBS_UP", 7, date(2025, 10, 24))])
    netflix.apply_decisions(resolved, {"7": "skip"})
    assert resolved[0].confidence == "skipped"
    assert not resolved[0].automatic


def test_a_choice_can_name_a_film_that_search_never_returned(fake_search):
    # TMDB's text search does not surface the Tamil "Youth" for an English
    # query, so a ruling has to be able to name a candidate outright.
    fake_search["Youth"] = [_hit(1, "Youth", 2015, votes=2268)]
    resolved = netflix.resolve([netflix.Rating("Youth", "THUMBS_WAY_UP", 82723855, date(2026, 4, 20))])
    netflix.apply_decisions(
        resolved, {"82723855": {"tmdb_id": 1542352, "kind": "movie", "language": "ta"}}
    )
    assert resolved[0].match["tmdb_id"] == 1542352
    assert resolved[0].match["language"] == "ta"


def test_titles_without_a_ruling_are_left_alone(fake_search):
    fake_search["Hunger"] = [
        _hit(1, "Hunger", 2008, popularity=4.7, votes=1257),
        _hit(2, "Hunger", 2023, popularity=3.3, votes=499),
    ]
    resolved = netflix.resolve([netflix.Rating("Hunger", "THUMBS_WAY_UP", 1, date(2026, 8, 15))])
    netflix.apply_decisions(resolved, {"999": "accept"})
    assert resolved[0].confidence == "ambiguous"


def test_an_unrecognised_ruling_is_an_error_not_a_silent_skip(fake_search):
    fake_search["X"] = [_hit(1, "X", 2020, votes=10)]
    resolved = netflix.resolve([netflix.Rating("X", "THUMBS_UP", 5, date(2025, 1, 1))])
    with pytest.raises(ValueError):
        netflix.apply_decisions(resolved, {"5": "maybe"})


# --- the import mechanics, lifted out of the CLI ----------------------------


def rating(title, netflix_id=None, thumbs="THUMBS_UP", rated_on=None):
    from datetime import date

    return netflix.Rating(
        title=title,
        thumbs=thumbs,
        netflix_id=netflix_id,
        rated_on=date.fromisoformat(rated_on) if rated_on else None,
    )


def test_dedupe_keeps_the_first_occurrence_because_later_pages_are_older():
    """Pages are grabbed by hand and overlap. The first occurrence of a title
    is the most recent verdict, so it is the one that must survive."""
    rows = [
        rating("Drishyam", 100, "THUMBS_WAY_UP"),
        rating("Drishyam", 100, "THUMBS_DOWN"),
    ]
    kept = netflix.dedupe(rows)
    assert len(kept) == 1
    assert kept[0].verdict == "love"


def test_dedupe_keys_on_the_netflix_id_as_well_as_the_title():
    """One title can be exported under more than one id, and two different
    titles can normalise alike."""
    rows = [rating("Hunger", 100), rating("Hunger", 200)]
    assert len(netflix.dedupe(rows)) == 2


def test_dedupe_normalises_before_comparing():
    rows = [rating("The Pope's Exorcist", 5), rating("the popes exorcist", 5)]
    assert len(netflix.dedupe(rows)) == 1


def test_dedupe_preserves_order():
    rows = [rating("A", 1), rating("B", 2), rating("A", 1), rating("C", 3)]
    assert [r.title for r in netflix.dedupe(rows)] == ["A", "B", "C"]


def test_review_payload_carries_the_netflix_id_so_rulings_can_be_replayed():
    """The decisions file is keyed on it. Without the id a ruling has to be
    retyped on every re-import of the same export."""
    res = netflix.Resolution(
        rating=rating("Hunger", 4242, rated_on="2024-03-01"),
        confidence="ambiguous",
        reason="several films share the title",
        match=None,
        alternatives=[{"tmdb_id": 1, "title": "Hunger"}],
    )
    payload = netflix.review_payload([res])[0]
    assert payload["netflix_id"] == 4242
    assert payload["rated_on"] == "2024-03-01"
    assert payload["confidence"] == "ambiguous"
    assert payload["alternatives"]


def test_bucket_groups_by_confidence():
    high = netflix.Resolution(rating=rating("A", 1), confidence="high", reason="", match={}, alternatives=[])
    low = netflix.Resolution(rating=rating("B", 2), confidence="low", reason="", match=None, alternatives=[])
    grouped = netflix.bucket([high, low, high])
    assert len(grouped["high"]) == 2
    assert len(grouped["low"]) == 1
    assert "ambiguous" not in grouped
