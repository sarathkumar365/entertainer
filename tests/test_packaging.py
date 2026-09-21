"""Guards against source files that exist on disk but not in the repository.

A bare `data/` in .gitignore matched `src/entertainer/data/` as well as the
intended top-level data directory, so the entire catalogue ingest layer —
about 1,100 lines — was never committed. Twenty commits referenced those files
in their messages and a fresh clone would not have run. Nothing in the test
suite noticed, because the tests ran against the working tree.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout


def _ignored(paths: list[str]) -> set[str]:
    """Which of these paths git would refuse to track."""
    if not paths:
        return set()
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", *paths],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    return set(result.stdout.split())


def _python_files(subdir: str) -> list[str]:
    return sorted(
        str(p.relative_to(REPO))
        for p in (REPO / subdir).rglob("*.py")
        if "__pycache__" not in p.parts
    )


@pytest.fixture(scope="module")
def in_git_checkout() -> bool:
    try:
        _git("rev-parse", "--git-dir")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("not a git checkout")
    return True


def test_no_source_file_is_ignored(in_git_checkout):
    """The actual failure mode: a file that cannot be committed even if added.

    Deliberately checks *ignored*, not *untracked*. A newly written file that
    has not been added yet is a normal working state; a file git will silently
    refuse is not.
    """
    files = _python_files("src")
    assert files
    ignored = _ignored(files)
    assert not ignored, f"source files git will not track: {sorted(ignored)}"


def test_no_test_file_is_ignored(in_git_checkout):
    files = _python_files("tests")
    assert files
    ignored = _ignored(files)
    assert not ignored, f"test files git will not track: {sorted(ignored)}"


def test_no_package_directory_is_ignored(in_git_checkout):
    """Catches the exact pattern that caused this: an ignored directory in src/."""
    packages = sorted(
        str(p.relative_to(REPO))
        for p in (REPO / "src").rglob("__init__.py")
        if "__pycache__" not in p.parts
    )
    assert packages
    ignored = _ignored(packages)
    assert not ignored, f"ignored package files: {sorted(ignored)}"


def test_secrets_and_personal_data_stay_ignored(in_git_checkout):
    """The pattern that caused the bug existed for a real reason; keep it working.

    `profile.jsonl` matters most: it is where `ent export` writes the user's
    entire verdict history by default, into the working directory, of a public
    repository.
    """
    should_be_ignored = [
        ".env",
        "data/entertainer.duckdb",
        "data/raw/imdb/title.basics.tsv.gz",
        "data/embeddings/fused.npz",
        "data/artifacts/taste.npz",
        "profile.jsonl",
        "profile-backup.jsonl",
        "my.profile.jsonl",
    ]
    ignored = _ignored(should_be_ignored)
    missing = [p for p in should_be_ignored if p not in ignored]
    assert not missing, f"these must never be committable: {missing}"


def test_declared_packages_match_what_is_on_disk():
    """A package added on disk but absent from the wheel config installs empty."""
    import tomllib

    config = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    declared = config["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    assert declared == ["src/entertainer"]
    # Everything under the declared root ships, so only confirm it exists.
    assert (REPO / "src" / "entertainer" / "__init__.py").exists()
