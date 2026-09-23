"""Verbatim CLI output, pinned.

The refactor moves rendering out of cli.py into a render package. Those moves
are meant to be no-ops, and "I read the diff and it looked fine" is not a way
to verify 114 console.print calls. Each case here captures the exact bytes a
command prints, so an extraction that changes a space, a column width or a
word fails loudly.

Policy: during a mechanical commit a moved golden means the commit is wrong.
Revert it; do not regenerate. Only a commit that deliberately changes what is
shown may move one, and it must move only the lines it claims to.

Regenerate deliberately with:

    ENTERTAINER_REGEN_GOLDEN=1 .venv/bin/python -m pytest tests/test_cli_golden.py

Two rules make the capture stable:

* COLUMNS is pinned. Console() takes no width, so rich reads the environment
  and table box-drawing follows the terminal. Unpinned goldens pass here and
  fail on a different machine.
* Only deterministic commands, so --strategy mean and never thompson.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
from test_cli import app_env, run, teach  # noqa: F401

GOLDEN = Path(__file__).parent / "golden"
REGEN = os.environ.get("ENTERTAINER_REGEN_GOLDEN") == "1"

_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2})?")
_ELAPSED = re.compile(r"\b\d+\.\d+s\b")


def scrub(text: str) -> str:
    """Remove the only two things that legitimately differ between runs."""
    text = _TIMESTAMP.sub("<timestamp>", text)
    return _ELAPSED.sub("<elapsed>", text)


SEED = [
    ("Kumbalangi Nights", "loved"),
    ("Jallikattu", "liked"),
    ("96", "loved"),
    ("Super Deluxe", "liked"),
    ("Kantara", "loved"),
    ("Morbius", "disliked"),
    ("Transformers", "disliked"),
    ("Parasite", "loved"),
    ("Memories of Murder", "liked"),
    ("Whiplash", "liked"),
]

# name, argv, stdin, how many verdicts to teach first
CASES = [
    ("stats_fresh", ("stats",), None, 0),
    ("audit_refuses_below_eight", ("audit",), None, 0),
    ("find_parasite", ("find", "Parasite"), None, 0),
    ("loved_nothing_matching", ("loved", "zzzzznotathing"), None, 0),
    ("loved_ambiguous_prompt", ("loved", "Drishyam"), "2\n", 0),
    ("stats_warmed", ("stats",), None, 10),
    ("history", ("history",), None, 10),
    ("taste", ("taste",), None, 10),
    ("recs_mean", ("recs", "-k", "6", "--strategy", "mean"), None, 10),
    ("similar", ("similar", "Parasite", "-k", "3"), None, 10),
    ("why", ("why", "Tumbbad"), None, 10),
    ("audit_full", ("audit",), None, 10),
]


@pytest.fixture(autouse=True)
def pinned_width(monkeypatch):
    """rich reads COLUMNS when the Console is constructed."""
    monkeypatch.setenv("COLUMNS", "100")
    monkeypatch.setenv("TERM", "dumb")


@pytest.mark.parametrize("name,argv,stdin,n_seed", CASES, ids=[c[0] for c in CASES])
def test_output_is_unchanged(app_env, name, argv, stdin, n_seed):  # noqa: F811
    cli, runner = app_env
    if n_seed:
        teach(cli, runner, SEED[:n_seed])

    result = run(cli, runner, *argv, stdin=stdin)
    actual = scrub(result.output)

    path = GOLDEN / f"{name}.txt"
    if REGEN:
        path.write_text(actual, encoding="utf-8")
        pytest.skip(f"regenerated {path.name}")

    assert path.exists(), f"missing golden {path.name}; regenerate deliberately"
    assert actual == path.read_text(encoding="utf-8")
