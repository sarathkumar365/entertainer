"""Publishing a build from one machine and pulling it on another.

Nothing here reaches GitHub: every `gh` call goes through a fake
``subprocess.run`` that records its arguments and serves release assets from
a local directory.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from entertainer import bundle, releases, store
from entertainer.config import PATHS

N = 40


def _use(tmp_path, monkeypatch, name) -> Path:
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("ENTERTAINER_DATA_DIR", str(root))
    PATHS.ensure()
    return root


def _populate(n=N, reverse=False):
    con = store.connect()
    for i in range(n):
        item_id = (n - 1 - i) if reverse else i
        con.execute(
            "INSERT INTO titles (item_id, imdb_id, kind, title, year, language) "
            "VALUES (?, ?, 'movie', ?, 2024, 'ml')",
            [item_id, f"tt{i:07d}", f"Film {i}"],
        )
    con.close()


def _space():
    np.save(PATHS.embeddings / "content_ids.npy", np.arange(N, dtype=np.int32))
    np.save(PATHS.embeddings / "content.npy", np.zeros((N, 8), dtype=np.float32))
    np.savez(PATHS.embeddings / "fused.npz", item_ids=np.arange(N))
    np.savez(PATHS.artifacts / "population_prior.npz", mean=np.zeros(4))


def _source(tmp_path, monkeypatch, name="source") -> Path:
    _use(tmp_path, monkeypatch, name)
    _populate()
    _space()
    manifests = PATHS.reports / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / "build-20260901T000000-aaaa.json").write_text(json.dumps({"id": "build-old"}))
    (manifests / "build-20260920T000000-bbbb.json").write_text(json.dumps({"id": "build-new"}))
    out = tmp_path / f"{name}.zip"
    bundle.export(out, include_space=True, include_encodings=True)
    return out


def _rewrite(src: Path, dst: Path, replace: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(src) as a, zipfile.ZipFile(dst, "w") as b:
        for item in a.namelist():
            b.writestr(item, replace.get(item, a.read(item)))
    return dst


def _count(sql):
    with store.session(read_only=True) as con:
        return con.execute(sql).fetchone()[0]


# --- manifest v2 ---------------------------------------------------------------


def test_manifest_v2_round_trips(tmp_path, monkeypatch):
    out = _source(tmp_path, monkeypatch)
    m = bundle.inspect(out)
    assert m["format"] == 2
    assert m["build_id"].startswith("cat-")
    assert m["source_build"] == "build-new"
    assert m["schema_version"] == store.SCHEMA_VERSION == 1 + len(store.MIGRATIONS)
    assert "imdb_id" in m["titles_columns"] and "keywords_at" in m["titles_columns"]
    assert m["encoder"]["dim"] == 8
    assert set(m["sha256"]) == set(m["contents"]) == {
        "titles.parquet", "fused.npz", "population_prior.npz", "content.npy", "content_ids.npy"
    }

    _use(tmp_path, monkeypatch, "target")
    result = bundle.restore(out)
    assert result["restored_titles"] == N
    assert result["warnings"] == []
    assert (PATHS.embeddings / "content.npy").exists()
    assert not list(PATHS.embeddings.glob(".*.incoming"))
    with store.session(read_only=True) as con:
        assert store.get_meta(con, bundle.BUILD_META) == m["build_id"]


def test_a_corrupt_file_is_rejected_before_the_database_is_touched(tmp_path, monkeypatch):
    out = _source(tmp_path, monkeypatch)
    bad = _rewrite(out, tmp_path / "bad.zip", {"fused.npz": b"not the space"})

    _use(tmp_path, monkeypatch, "target")
    _populate(n=5)
    with store.session() as con:
        store.log_event(con, 1, "rate", 9.0)
    np.savez(PATHS.embeddings / "fused.npz", item_ids=np.arange(5))
    before = PATHS.catalog_db.stat()
    fused_before = (PATHS.embeddings / "fused.npz").read_bytes()

    with pytest.raises(bundle.BundleError, match="fused.npz does not match"):
        bundle.restore(bad, overwrite=True)

    after = PATHS.catalog_db.stat()
    assert (after.st_mtime_ns, after.st_size) == (before.st_mtime_ns, before.st_size)
    assert (PATHS.embeddings / "fused.npz").read_bytes() == fused_before
    assert _count("SELECT count(*) FROM titles") == 5
    assert not (PATHS.root / ".bundle-restore").exists()


def test_a_format_1_bundle_still_imports_with_a_warning(tmp_path, monkeypatch):
    out = _source(tmp_path, monkeypatch)
    old = _rewrite(out, tmp_path / "old.zip", {
        bundle.MANIFEST: json.dumps({"format": 1, "titles": N, "contents": ["titles.parquet"]}).encode()
    })
    _use(tmp_path, monkeypatch, "target")
    result = bundle.restore(old)
    assert result["restored_titles"] == N
    assert "no checksums" in result["warnings"][0]


def test_a_newer_schema_is_refused(tmp_path, monkeypatch):
    out = _source(tmp_path, monkeypatch)
    m = bundle.inspect(out)
    with pytest.raises(bundle.BundleError, match="git pull"):
        bundle.check_compatible({**m, "schema_version": store.SCHEMA_VERSION + 1})
    with pytest.raises(bundle.BundleError, match="does not know"):
        bundle.check_compatible({**m, "titles_columns": [*m["titles_columns"], "mood"]})
    with pytest.raises(bundle.BundleError, match="format 3"):
        bundle.check_compatible({**m, "format": 3})

    newer = _rewrite(out, tmp_path / "newer.zip", {
        bundle.MANIFEST: json.dumps({**m, "schema_version": store.SCHEMA_VERSION + 1}).encode()
    })
    _use(tmp_path, monkeypatch, "target")
    with pytest.raises(bundle.BundleError):
        bundle.restore(newer)
    assert not PATHS.catalog_db.exists()


# --- a fake gh -----------------------------------------------------------------


class FakeGh:
    """Answers the handful of `gh` invocations releases.py makes."""

    def __init__(self, visibility="PRIVATE", assets: dict[str, Path] | None = None, authed=True):
        self.calls: list[list[str]] = []
        self.visibility = visibility
        self.assets = assets or {}
        self.authed = authed

    def release_json(self, tag="build-20260920-0000"):
        return {
            "tag_name": tag,
            "published_at": "2026-09-20T00:00:00Z",
            "assets": [{"name": n, "size": p.stat().st_size} for n, p in self.assets.items()],
        }

    def __call__(self, args, **_kw):
        self.calls.append(list(args))
        assert args[0] == "gh"
        rest = args[1:]
        out, code = "", 0
        if rest[:2] == ["auth", "status"]:
            code = 0 if self.authed else 1
        elif rest[:2] == ["repo", "view"]:
            out = json.dumps({"visibility": self.visibility})
        elif rest[:2] == ["release", "create"]:
            pass
        elif rest[0] == "api":
            path = rest[1]
            out = json.dumps([self.release_json()] if path.split("?")[0].endswith("/releases") else self.release_json())
        elif rest[:2] == ["release", "download"]:
            name = rest[rest.index("--pattern") + 1]
            directory = Path(rest[rest.index("--dir") + 1])
            shutil.copy(self.assets[name], directory / name)
        else:
            raise AssertionError(f"unexpected gh call: {args}")
        return subprocess.CompletedProcess(args, code, stdout=out, stderr="" if code == 0 else "nope")


@pytest.fixture
def gh(monkeypatch):
    fake = FakeGh()
    monkeypatch.setattr(releases.subprocess, "run", fake)
    monkeypatch.setattr(releases.shutil, "which", lambda name: "/usr/bin/gh")
    monkeypatch.setattr(releases, "runtime_dirs", lambda: [PATHS.root / "runtime"])
    return fake


def test_publish_refuses_a_public_repository(tmp_path, monkeypatch, gh):
    _source(tmp_path, monkeypatch)
    gh.visibility = "PUBLIC"
    with pytest.raises(releases.ReleaseError, match="public, not private"):
        releases.publish("someone/entertainer")
    assert not any(c[1:3] == ["release", "create"] for c in gh.calls)


def test_publish_needs_gh_installed_and_logged_in(tmp_path, monkeypatch, gh):
    gh.authed = False
    with pytest.raises(releases.ReleaseError, match="gh auth login"):
        releases.publish("me/builds")
    monkeypatch.setattr(releases.shutil, "which", lambda name: None)
    with pytest.raises(releases.ReleaseError, match="not installed"):
        releases.publish("me/builds")


def test_publish_creates_a_release_with_the_bundle_and_its_manifest(tmp_path, monkeypatch, gh):
    _source(tmp_path, monkeypatch)
    done = releases.publish("me/builds", tag="build-20260920-0000")
    create = next(c for c in gh.calls if c[1:3] == ["release", "create"])
    assert create[3] == "build-20260920-0000"
    assert Path(create[4]).name == "entertainer-build-20260920-0000.zip"
    assert Path(create[5]).name == "bundle.json"
    assert create[create.index("--repo") + 1] == "me/builds"
    assert "--title" in create and "--notes" in create
    assert "content.npy" in done.info.contents  # encodings travel in a release
    assert gh.calls.index(["gh", "repo", "view", "me/builds", "--json", "visibility"]) < gh.calls.index(create)
    with store.session(read_only=True) as con:
        assert store.get_meta(con, bundle.BUILD_META) == done.info.manifest["build_id"]


def test_repo_defaults_to_the_private_builds_repo(monkeypatch):
    monkeypatch.delenv(releases.REPO_ENV, raising=False)
    assert releases.releases_repo() == "sarathkumar365/entertainer-builds"
    monkeypatch.setenv(releases.REPO_ENV, "me/elsewhere")
    assert releases.releases_repo() == "me/elsewhere"
    assert releases.releases_repo("x/y") == "x/y"
    assert releases.default_tag(0) == "build-19700101-0000"


# --- pull ----------------------------------------------------------------------


def _serve(gh, archive: Path, tmp_path):
    manifest = tmp_path / "served-bundle.json"
    manifest.write_text(json.dumps(bundle.inspect(archive)))
    gh.assets = {"entertainer-build-20260920-0000.zip": archive, "bundle.json": manifest}


def test_pull_keeps_and_remaps_local_verdicts(tmp_path, monkeypatch, gh):
    archive = _source(tmp_path, monkeypatch)
    _serve(gh, archive, tmp_path)

    # Same titles, opposite item ids: every verdict must move.
    _use(tmp_path, monkeypatch, "laptop")
    _populate(reverse=True)
    with store.session() as con:
        store.log_event(con, 39, "rate", 9.0)            # tt0000000
        store.log_event(con, 30, "skip")                 # tt0000009
        store.log_impressions(con, "s1", [(35, 0, 1.0, 0.5, False)], "p")  # tt0000004
        con.execute(
            "INSERT INTO validation_batches VALUES ('b1', now(), '[38, 37]', '[37, 38]', '[38]', NULL)"
        )
        con.execute(
            "INSERT INTO validation_cases (case_id, batch_id, item_id, full_score, full_std, "
            "full_like_prob, ridge_score, ridge_like_prob) VALUES ('c1', 'b1', 38, 0, 0, 0, 0, 0)"
        )
        store.set_last_slate(con, [39, 30])

    work = tmp_path / "work"
    work.mkdir()
    plan = releases.plan_pull("me/builds", "latest", work)
    assert (plan.local_titles, plan.local_events, plan.local_build) == (N, 2, None)
    assert plan.manifest["build_id"].startswith("cat-") and not plan.up_to_date
    result = releases.apply_pull(plan, work)
    assert result["preserved_events"] == 2

    with store.session(read_only=True) as con:
        verdicts = con.execute(
            "SELECT t.imdb_id, e.kind, e.item_id FROM events e JOIN titles t USING (item_id) ORDER BY 1"
        ).fetchall()
        assert verdicts == [("tt0000000", "rate", 0), ("tt0000009", "skip", 9)]
        assert con.execute("SELECT item_id FROM impressions").fetchone()[0] == 4
        assert con.execute("SELECT item_id FROM validation_cases").fetchone()[0] == 1
        batch = con.execute("SELECT item_ids, full_ranking FROM validation_batches").fetchone()
        assert (json.loads(batch[0]), json.loads(batch[1])) == ([1, 2], [2, 1])
        assert store.last_slate(con) == [0, 9]
        assert store.get_meta(con, bundle.BUILD_META) == plan.manifest["build_id"]

    plan = releases.plan_pull("me/builds", "latest", work)
    assert plan.up_to_date


def test_pull_refuses_when_a_verdict_would_be_orphaned(tmp_path, monkeypatch, gh):
    archive = _source(tmp_path, monkeypatch)
    _serve(gh, archive, tmp_path)
    _use(tmp_path, monkeypatch, "laptop")
    _populate()
    with store.session() as con:
        con.execute("INSERT INTO titles (item_id, imdb_id, title) VALUES (500, 'tt9999999', 'Local only')")
        store.log_event(con, 500, "rate", 8.0)
    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(bundle.BundleError, match="cannot preserve 1"):
        releases.apply_pull(releases.plan_pull("me/builds", "latest", work), work)
    assert _count("SELECT count(*) FROM titles") == N + 1


def test_pull_rejects_an_archive_that_disagrees_with_its_manifest(tmp_path, monkeypatch, gh):
    archive = _source(tmp_path, monkeypatch)
    _serve(gh, archive, tmp_path)
    m = json.loads(gh.assets["bundle.json"].read_text())
    gh.assets["bundle.json"].write_text(json.dumps({**m, "build_id": "build-other"}))
    _use(tmp_path, monkeypatch, "laptop")
    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(releases.ReleaseError, match="disagree"):
        releases.apply_pull(releases.plan_pull("me/builds", "latest", work), work)
    assert not PATHS.catalog_db.exists()


def test_pull_refuses_while_the_app_is_running(tmp_path, monkeypatch, gh):
    from entertainer.cli import app

    _use(tmp_path, monkeypatch, "laptop")
    runtime = PATHS.root / "runtime"
    runtime.mkdir()
    (runtime / "app.pid").write_text(str(os.getpid()))

    result = CliRunner().invoke(app, ["pull", "--repo", "me/builds", "--yes"])
    assert result.exit_code == 1
    assert "entertainer stop" in result.output
    assert not any(c[1] == "api" for c in gh.calls)

    # A stale pid file from a crashed app is not a reason to refuse.
    dead = subprocess.Popen(["true"])
    dead.wait()
    (runtime / "app.pid").write_text(str(dead.pid))
    releases.require_app_stopped()


def test_release_list_shows_tags_and_sizes(tmp_path, monkeypatch, gh):
    from entertainer.cli import app

    archive = _source(tmp_path, monkeypatch)
    _serve(gh, archive, tmp_path)
    result = CliRunner().invoke(app, ["release", "list", "--repo", "me/builds"])
    assert result.exit_code == 0, result.output
    assert "build-20260920-0000" in result.output
    assert "MB" in result.output


# --- build identity -----------------------------------------------------------


def test_the_build_id_names_the_content_not_the_last_local_build(tmp_path, monkeypatch):
    first = bundle.inspect(_source(tmp_path, monkeypatch))
    again = bundle.inspect(bundle.export(tmp_path / "again.zip", True, True).path)
    assert again["build_id"] == first["build_id"]

    # A partial rebuild (`data fuse`) changes the space but writes no manifest.
    np.savez(PATHS.embeddings / "fused.npz", item_ids=np.arange(N) + 1)
    changed = bundle.inspect(bundle.export(tmp_path / "changed.zip", True, True).path)
    assert changed["source_build"] == first["source_build"]
    assert changed["build_id"] != first["build_id"]


def test_a_local_rebuild_forgets_which_release_was_pulled(tmp_path, monkeypatch):
    from entertainer.data import catalog

    _use(tmp_path, monkeypatch, "laptop")
    with store.session() as con:
        store.set_meta(con, bundle.BUILD_META, "cat-abc")
    catalog.prune_by_language()
    assert _count("SELECT count(*) FROM meta WHERE key = 'catalogue_build'") == 0


def test_a_format_1_import_forgets_which_release_was_pulled(tmp_path, monkeypatch):
    archive = _source(tmp_path, monkeypatch)
    m = bundle.inspect(archive)
    legacy = {"format": 1, "titles": m["titles"], "contents": m["contents"]}
    old = _rewrite(archive, tmp_path / "legacy.zip", {bundle.MANIFEST: json.dumps(legacy).encode()})

    _use(tmp_path, monkeypatch, "laptop")
    with store.session() as con:
        store.set_meta(con, bundle.BUILD_META, "cat-abc")
    bundle.restore(old)
    assert _count("SELECT count(*) FROM meta WHERE key = 'catalogue_build'") == 0
