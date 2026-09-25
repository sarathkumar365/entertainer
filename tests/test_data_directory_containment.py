"""The suite must write only inside its own temporary data directory.

This is a regression guard for a bug that had already caused real damage: the
fixtures used to keep the test data directory isolated by reloading a list of
modules after setting ``ENTERTAINER_DATA_DIR``. Fifteen modules do
``from .config import PATHS``, which binds the *object*, so reloading
``config`` rebinds only its own name. ``models.taste`` was never in the list,
and ``Engine.fit(save=True)`` calls ``TasteModel.to_npz()``, which resolves
``PATHS.artifacts / "taste.npz"`` through that stale binding.

The result: running the suite replaced the real 1.9 MB taste model, fitted on
the user's actual verdicts, with a 4.6 KB model fitted on synthetic test
titles. Silently, and on every run.

The fix was to resolve ``Paths.root`` on each access instead of at import, so
no reload list is needed at all. These two tests pin both halves of that: the
behaviour, and the invariant it depends on.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"


def test_changing_the_data_directory_moves_every_module_that_binds_paths(monkeypatch, tmp_path):
    """The invariant the whole fix rests on.

    These modules are imported here first, so they bind ``PATHS`` while it
    still points at the real data directory — exactly the state a full-suite
    run leaves them in. Changing the environment afterwards must move all of
    them. Under the old frozen ``Paths(_data_dir())`` it moved none, and only
    the handful named in each fixture's reload list appeared to work.
    """
    from entertainer import bundle, config, engine, manifests, pipeline, store
    from entertainer.data import catalog, download, imdb
    from entertainer.models import cf, encoder, fusion, taste

    moved = tmp_path / "moved"
    monkeypatch.setenv("ENTERTAINER_DATA_DIR", str(moved))

    binders = {
        "config": config,
        "store": store,
        "engine": engine,
        "bundle": bundle,
        "manifests": manifests,
        "pipeline": pipeline,
        "catalog": catalog,
        "download": download,
        "imdb": imdb,
        "cf": cf,
        "encoder": encoder,
        "fusion": fusion,
        "taste": taste,
    }
    stale = {name: str(mod.PATHS.root) for name, mod in binders.items() if mod.PATHS.root != moved}
    assert not stale, f"modules still pointing at the old data directory: {stale}"

    # The specific write that destroyed the real model.
    assert taste.PATHS.artifacts / "taste.npz" == moved / "artifacts" / "taste.npz"


def test_no_module_resolves_a_data_path_at_import_time():
    """``Paths.root`` is resolved lazily, which only helps if nothing captures
    a derived path at module scope. A module-level ``PATHS.artifacts`` would
    freeze the directory at import and reintroduce the bug."""
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for sub in ast.walk(node):
                if (
                    isinstance(sub, ast.Attribute)
                    and isinstance(sub.value, ast.Name)
                    and sub.value.id == "PATHS"
                ):
                    offenders.append(f"{path.relative_to(SRC)}:{sub.lineno} PATHS.{sub.attr}")
    assert not offenders, "paths resolved at import time:\n  " + "\n  ".join(offenders)
