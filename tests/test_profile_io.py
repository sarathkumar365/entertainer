"""The verdict-history file formats.

Extracted from `ent export`, `ent import` and `ent bulk`, where the parsing
and serialisation lived in the command bodies. These formats are the only
record of a taste profile that survives a catalogue rebuild, so they are
worth testing directly rather than through a CLI invocation.
"""

from __future__ import annotations

import json

import pytest
from test_cli import app_env  # noqa: F401

from entertainer import profile_io, store
from entertainer.errors import EntertainerError


def test_bulk_defaults_the_verdict_when_a_line_gives_none():
    got = profile_io.parse_bulk_lines("Kumbalangi Nights\n", default_verdict="love")
    assert got == [profile_io.BulkLine(title="Kumbalangi Nights", verdict="love")]


def test_bulk_reads_an_explicit_verdict():
    got = profile_io.parse_bulk_lines("Morbius | hate\n")
    assert got == [profile_io.BulkLine(title="Morbius", verdict="hate")]


def test_bulk_splits_on_the_last_pipe_so_titles_may_contain_one():
    """`split` on the first pipe would truncate the title."""
    got = profile_io.parse_bulk_lines("Tinker Tailor Soldier | Spy | love\n")
    assert got == [profile_io.BulkLine(title="Tinker Tailor Soldier | Spy", verdict="love")]


def test_bulk_skips_blank_lines_and_comments():
    assert profile_io.parse_bulk_lines("\n# a note\n\n  \nDark\n") == [
        profile_io.BulkLine(title="Dark", verdict="like")
    ]


def test_export_then_import_round_trips(app_env, tmp_path):  # noqa: F811
    cli, runner = app_env
    from test_cli import run, teach

    teach(cli, runner, [("Kumbalangi Nights", "loved"), ("Morbius", "disliked")])

    path = tmp_path / "profile.jsonl"
    with store.session(read_only=True) as con:
        written = profile_io.export_events(con, path)
    assert written == 2

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert {r["imdb_id"] for r in records} == {"tt10000001", "tt10000009"}
    assert all(r["kind"] == "rate" for r in records)

    # Re-importing an unchanged export must be a no-op, not a doubling.
    with store.session() as con:
        counts = profile_io.import_events(con, path)
    assert counts.imported == 0
    assert counts.duplicates == 2
    assert counts.missing == 0
    assert run(cli, runner, "history").exit_code == 0


def test_import_counts_titles_the_catalogue_does_not_have(app_env, tmp_path):  # noqa: F811
    path = tmp_path / "other.jsonl"
    path.write_text(
        json.dumps(
            {"imdb_id": "tt99999999", "kind": "rate", "value": 10.0,
             "ts": "2024-01-01 00:00:00", "source": "import", "context": {}}
        )
        + "\n"
    )
    with store.session() as con:
        counts = profile_io.import_events(con, path)
    assert counts == profile_io.ImportCounts(imported=0, duplicates=0, missing=1)


def test_import_names_the_offending_line_on_bad_json(app_env, tmp_path):  # noqa: F811
    path = tmp_path / "broken.jsonl"
    good = json.dumps(
        {"imdb_id": "tt10000001", "kind": "rate", "value": 10.0,
         "ts": "2024-01-01 00:00:00", "source": "import", "context": {}}
    )
    path.write_text(good + "\nnot json at all\n")
    with store.session() as con, pytest.raises(EntertainerError) as exc:
        profile_io.import_events(con, path)
    assert "broken.jsonl:2" in str(exc.value)
    assert "not valid JSON" in str(exc.value)


def test_import_names_the_offending_line_on_a_missing_field(app_env, tmp_path):  # noqa: F811
    """A record without `kind` cannot be replayed; say which line it was."""
    path = tmp_path / "partial.jsonl"
    path.write_text(json.dumps({"imdb_id": "tt10000001"}) + "\n")
    with store.session() as con, pytest.raises(EntertainerError) as exc:
        profile_io.import_events(con, path)
    assert "partial.jsonl:1" in str(exc.value)
