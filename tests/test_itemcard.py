from entertainer.models.itemcard import build_card

BASE = {
    "title": "Kumbalangi Nights",
    "original_title": "Kumbalangi Nights",
    "year": 2019,
    "kind": "movie",
    "language": "ml",
    "runtime": 135,
    "genres": ["Drama", "Comedy"],
    "keywords": ["dysfunctional family", "brothers"],
    "directors": ["Madhu C. Narayanan"],
    "cast_names": ["Shane Nigam", "Fahadh Faasil"],
    "overview": "Four brothers in a Kerala fishing village.",
}


def test_card_is_prose_and_contains_the_signal():
    card = build_card(BASE)
    assert "Malayalam-language film" in card
    assert "Madhu C. Narayanan" in card
    assert "Four brothers" in card
    assert "135 min" in card


def test_card_survives_missing_fields():
    card = build_card({"title": "Unknown Thing"})
    assert card.startswith("Unknown Thing")


def test_original_title_included_when_different():
    row = dict(BASE, original_title="കുമ്പളങ്ങി നൈറ്റ്സ്")
    assert "കുമ്പളങ്ങി" in build_card(row)


def test_overview_is_truncated():
    row = dict(BASE, overview="x" * 5000)
    assert len(build_card(row)) < 2000
