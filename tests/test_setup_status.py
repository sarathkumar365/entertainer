import json
import subprocess
import sys
import types

import pytest
from fastapi.testclient import TestClient

from entertainer import setup_status as ss
from entertainer import store
from entertainer.bundle import BUILD_META
from entertainer.config import PATHS


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTERTAINER_DATA_DIR", str(tmp_path))
    PATHS.ensure()
    return tmp_path


def _fake_run(monkeypatch, responses: dict[tuple, tuple[int, str]]):
    """Answer `_run(*cmd)` from the longest matching command prefix."""
    calls = []

    def run(*cmd):
        calls.append(cmd)
        for prefix in sorted(responses, key=len, reverse=True):
            if cmd[: len(prefix)] == prefix:
                code, out = responses[prefix]
                return subprocess.CompletedProcess(cmd, code, stdout=out, stderr="")
        return None

    monkeypatch.setattr(ss, "_run", run)
    return calls


def _by_id(checks):
    return {c.id: c for c in checks}


# --- schedule ------------------------------------------------------------------


def test_a_publishing_cron_job_is_found_with_its_schedule(monkeypatch):
    _fake_run(monkeypatch, {("crontab", "-l"): (0, "# nightly\n0 3 * * 0  cd /x && ent setup && ent release publish\n")})
    check = ss._cron(ss.BUILDER)
    assert check.status == ss.OK
    assert check.data["schedule"] == "0 3 * * 0"


def test_a_build_that_never_publishes_is_flagged(monkeypatch):
    _fake_run(monkeypatch, {("crontab", "-l"): (0, "0 3 * * * /x/.venv/bin/ent setup\n")})
    assert ss._cron(ss.BUILDER).status == ss.WARN


def test_no_cron_matters_only_on_the_builder(monkeypatch):
    _fake_run(monkeypatch, {("crontab", "-l"): (1, "")})
    builder = ss._cron(ss.BUILDER)
    assert builder.status == ss.MISSING
    assert "release publish" in builder.fix and "| crontab -" in builder.fix
    assert ss._cron(ss.PULLER).status == ss.OK


# --- releases ------------------------------------------------------------------


def _gh_ok(monkeypatch, visibility="PRIVATE", latest=None):
    monkeypatch.setattr(ss.shutil, "which", lambda name: "/usr/bin/gh")
    return _fake_run(monkeypatch, {
        ("gh", "auth"): (0, ""),
        ("gh", "repo", "view"): (0, json.dumps({"visibility": visibility, "url": "u"})),
        # releases/latest answers 404 when nothing is published.
        ("gh", "api"): (0, json.dumps(latest)) if latest else (1, ""),
    })


LATEST = {"tag_name": "build-1", "published_at": "2026-09-23T00:00:00Z",
          "body": "40 titles · build cat-abc123 · schema 2", "assets": [{"size": 5_000_000}]}


def test_missing_gh_says_how_to_install_it(monkeypatch):
    monkeypatch.setattr(ss.shutil, "which", lambda name: None)
    (check,) = ss._gh("me/builds")
    assert check.status == ss.MISSING and "gh" in check.fix


def test_a_public_builds_repo_is_a_warning(monkeypatch):
    _gh_ok(monkeypatch, visibility="PUBLIC")
    checks = _by_id(ss._gh("me/builds"))
    assert checks["repo"].status == ss.WARN
    assert "--visibility private" in checks["repo"].fix


def test_nothing_published_yet(monkeypatch):
    _gh_ok(monkeypatch)
    assert _by_id(ss._gh("me/builds"))["latest"].status == ss.MISSING


def test_this_machine_is_up_to_date_when_it_pulled_the_latest(monkeypatch, data_dir):
    calls = _gh_ok(monkeypatch, latest=LATEST)
    with store.session() as con:
        store.set_meta(con, BUILD_META, "cat-abc123")
    checks = _by_id(ss._gh("me/builds"))
    assert checks["latest"].data["build_id"] == "cat-abc123"
    assert checks["pulled"].status == ss.OK
    assert ("gh", "api", "repos/me/builds/releases/latest") in calls

    with store.session() as con:
        store.set_meta(con, BUILD_META, "cat-old")
    pulled = _by_id(ss._gh("me/builds"))["pulled"]
    assert pulled.status == ss.WARN and "pull" in pulled.fix


# --- this machine ----------------------------------------------------------------


def test_an_old_nvidia_driver_is_flagged(monkeypatch):
    from entertainer.models import encoder

    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setattr(encoder, "_device", lambda: "cuda")
    _fake_run(monkeypatch, {("nvidia-smi",): (0, "RTX 4090, 550.54, 24564\n")})
    assert ss._device(ss.BUILDER).status == ss.WARN
    _fake_run(monkeypatch, {("nvidia-smi",): (0, "RTX 4090, 575.51, 24564\n")})
    assert ss._device(ss.BUILDER).status == ss.OK


def test_the_role_can_be_overridden(monkeypatch):
    monkeypatch.setenv(ss.ROLE_ENV, "builder")
    assert ss.role() == ss.BUILDER
    monkeypatch.setenv(ss.ROLE_ENV, "puller")
    assert ss.role() == ss.PULLER


def test_collect_reports_the_worst_status(monkeypatch, data_dir):
    monkeypatch.setenv(ss.ROLE_ENV, "puller")
    monkeypatch.setattr(ss.shutil, "which", lambda name: None)
    _fake_run(monkeypatch, {("crontab", "-l"): (1, "")})
    report = ss.collect("me/builds")
    assert report["role"] == ss.PULLER
    assert report["overall"] == ss.MISSING  # no model files, no gh
    assert {c["id"] for c in report["checks"]} >= {"device", "artefacts", "gh", "cron"}


# --- Build Studio ----------------------------------------------------------------


def test_studio_serves_the_report_and_caches_it(monkeypatch, data_dir):
    from entertainer.web.studio import create_studio_app

    calls = []
    monkeypatch.setattr(ss, "collect", lambda repo=None: calls.append(1) or {"overall": "ok"})
    client = TestClient(create_studio_app())
    assert client.get("/api/setup").json() == {"overall": "ok"}
    client.get("/api/setup")
    assert len(calls) == 1
    client.get("/api/setup?refresh=true")
    assert len(calls) == 2


def test_checking_never_creates_a_catalogue(monkeypatch, data_dir):
    _gh_ok(monkeypatch, latest=LATEST)
    pulled = _by_id(ss._gh("me/builds"))["pulled"]
    assert pulled.status == ss.MISSING
    assert not PATHS.catalog_db.exists()


def test_a_builder_is_asked_to_publish_not_to_pull(monkeypatch, data_dir):
    _gh_ok(monkeypatch, latest=LATEST)
    with store.session():
        pass  # a catalogue built here: no published-build record
    pulled = _by_id(ss._gh("me/builds", ss.BUILDER))["pulled"]
    assert pulled.status == ss.WARN and pulled.fix == "ent release publish"

    with store.session() as con:
        store.set_meta(con, BUILD_META, "cat-abc123")
    assert _by_id(ss._gh("me/builds", ss.BUILDER))["pulled"].status == ss.OK


@pytest.mark.parametrize("line", ["NVIDIA A100, PCIe, 575.51, [N/A]\n", "weird\n"])
def test_odd_nvidia_smi_output_does_not_break_the_report(monkeypatch, line):
    from entertainer.models import encoder

    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setattr(encoder, "_device", lambda: "cuda")
    _fake_run(monkeypatch, {("nvidia-smi",): (0, line)})
    assert ss._device(ss.BUILDER).status == ss.OK


def test_the_suggested_cron_line_creates_its_log_dir_and_logs_both_steps():
    line = ss.suggested_cron()
    assert line.index("mkdir -p") < line.index("setup")
    assert "{" in line and "; }" in line and line.endswith("2>&1")
