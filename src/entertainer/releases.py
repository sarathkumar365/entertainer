"""Publish a built catalogue from one machine and pull it on another.

The build wants a GPU and hours; using the result wants neither. So the
machine that builds publishes a full bundle as a GitHub release, and every
other machine pulls the latest one instead of building.

Releases go to a separate repository, never the source one: the source is
public, and the bundle is IMDb and TMDB derived data whose terms do not allow
redistributing it. Publishing therefore refuses any repository that GitHub
does not report as private.

All GitHub traffic goes through the `gh` CLI, so credentials are whatever
`gh auth login` set up and never pass through this code.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import duckdb

from . import bundle, store
from .config import PATHS, REPO_ROOT
from .errors import EntertainerError

DEFAULT_REPO = "sarathkumar365/entertainer-builds"
REPO_ENV = "ENTERTAINER_RELEASES_REPO"


class ReleaseError(EntertainerError):
    """Publishing or pulling could not go ahead."""


def releases_repo(override: str | None = None) -> str:
    return override or os.environ.get(REPO_ENV) or DEFAULT_REPO


def default_tag(now: float | None = None) -> str:
    return time.strftime("build-%Y%m%d-%H%M", time.gmtime(now))


# --- gh ---------------------------------------------------------------------


def _gh(*args: str) -> str:
    try:
        result = subprocess.run(["gh", *args], capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise ReleaseError(_NO_GH) from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise ReleaseError(f"`gh {args[0]} {args[1] if len(args) > 1 else ''}` failed: {detail}")
    return result.stdout


_NO_GH = "the GitHub CLI `gh` is not installed; see https://cli.github.com, then run `gh auth login`"


def require_gh() -> None:
    if shutil.which("gh") is None:
        raise ReleaseError(_NO_GH)
    result = subprocess.run(
        ["gh", "auth", "status", "--hostname", "github.com"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise ReleaseError("`gh` is not logged in to github.com; run `gh auth login` first")


def require_private(repo: str) -> None:
    visibility = json.loads(_gh("repo", "view", repo, "--json", "visibility")).get("visibility")
    if visibility != "PRIVATE":
        raise ReleaseError(
            f"{repo} is {str(visibility).lower()}, not private. The bundle is IMDb and TMDB "
            f"derived data and must not be published openly; point {REPO_ENV} at a private "
            "repository"
        )


# --- publish ----------------------------------------------------------------


@dataclass
class Published:
    repo: str
    tag: str
    info: bundle.BundleInfo


def publish(repo: str, tag: str | None = None, notes: str | None = None) -> Published:
    """Export a full bundle and attach it, and its manifest, to a new release."""
    require_gh()
    require_private(repo)
    tag = tag or default_tag()
    PATHS.ensure()
    with tempfile.TemporaryDirectory(dir=PATHS.root, prefix=".release-") as work:
        zip_path = Path(work) / f"entertainer-{tag}.zip"
        info = bundle.export(zip_path, include_space=True, include_encodings=True)
        # Published on its own too, so a machine can decide whether to pull
        # without first downloading the archive.
        manifest_path = Path(work) / bundle.MANIFEST
        manifest_path.write_text(json.dumps(info.manifest, indent=2), encoding="utf-8")
        m = info.manifest
        _gh(
            "release", "create", tag, str(zip_path), str(manifest_path),
            "--repo", repo,
            "--title", f"Catalogue {tag}",
            "--notes", notes or (
                f"{m['titles']:,} titles · build {m['build_id']} · schema {m['schema_version']}\n"
                f"created {m['created_at']} · {', '.join(m['contents'])}"
            ),
        )
    # This catalogue is now that release, so `ent pull` and the setup checks
    # on the builder see it as current rather than as an unpublished build.
    with store.session() as con:
        store.set_meta(con, bundle.BUILD_META, info.manifest["build_id"])
    return Published(repo=repo, tag=tag, info=info)


# --- list and pull ----------------------------------------------------------


def list_releases(repo: str, limit: int = 30) -> list[dict]:
    require_gh()
    return json.loads(_gh("api", f"repos/{repo}/releases?per_page={int(limit)}"))


def get_release(repo: str, tag: str = "latest") -> dict:
    require_gh()
    path = f"repos/{repo}/releases/latest" if tag == "latest" else f"repos/{repo}/releases/tags/{tag}"
    try:
        return json.loads(_gh("api", path))
    except ReleaseError as exc:
        raise ReleaseError(f"no release {tag!r} in {repo} (or no access to it): {exc}") from exc


def _asset(release: dict, predicate) -> dict | None:
    return next((a for a in release.get("assets", []) if predicate(a["name"])), None)


def _download(repo: str, tag: str, asset: dict, directory: Path) -> Path:
    _gh(
        "release", "download", tag, "--repo", repo,
        "--pattern", asset["name"], "--dir", str(directory), "--clobber",
    )
    path = directory / asset["name"]
    if not path.exists():
        raise ReleaseError(f"download of {asset['name']} produced no file")
    # GitHub records a digest for assets uploaded since mid-2025.
    digest = asset.get("digest") or ""
    if digest.startswith("sha256:"):
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        if h.hexdigest() != digest.removeprefix("sha256:"):
            raise ReleaseError(f"{asset['name']} does not match GitHub's digest; download again")
    return path


@dataclass
class PullPlan:
    repo: str
    tag: str
    manifest: dict
    archive: dict
    local_titles: int = 0
    local_events: int = 0
    local_build: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def up_to_date(self) -> bool:
        return bool(self.local_build) and self.local_build == self.manifest.get("build_id")


def plan_pull(repo: str, tag: str, work: Path) -> PullPlan:
    """Fetch only the release's manifest and compare it with this machine."""
    release = get_release(repo, tag)
    archive = _asset(release, lambda n: n.endswith(".zip"))
    described = _asset(release, lambda n: n == bundle.MANIFEST)
    if archive is None or described is None:
        raise ReleaseError(f"release {release.get('tag_name')} has no bundle and manifest attached")
    manifest = json.loads(
        _download(repo, release["tag_name"], described, work).read_text(encoding="utf-8")
    )
    plan = PullPlan(
        repo=repo, tag=release["tag_name"], manifest=manifest, archive=archive,
        warnings=bundle.check_compatible(manifest),
    )
    if PATHS.catalog_db.exists():
        with store.session(read_only=True) as con:
            counts = store.counts(con)
            plan.local_build = store.get_meta(con, bundle.BUILD_META)
        plan.local_titles, plan.local_events = counts["titles"], counts["events"]
    return plan


def apply_pull(plan: PullPlan, work: Path) -> dict:
    """Download the archive, check it is the one described, and import it.

    Import keeps every local event, impression and validation case, moved onto
    the new catalogue's ids by IMDb id; it refuses outright if any of them
    names a title the new catalogue lacks.
    """
    require_app_stopped()
    archive = _download(plan.repo, plan.tag, plan.archive, work)
    inner = bundle.inspect(archive)
    if inner.get("build_id") != plan.manifest.get("build_id") or inner.get("sha256") != plan.manifest.get("sha256"):
        raise ReleaseError("the release's archive and manifest disagree; refusing to import it")
    try:
        return bundle.restore(archive, overwrite=True)
    except duckdb.IOException as exc:
        # Another process (a build, an unmanaged `ent rate`) holds the file;
        # the pid check only knows about the helper's own services.
        raise ReleaseError(
            f"the catalogue is open in another process, so nothing was changed: {exc}"
        ) from exc


# --- the managed app ----------------------------------------------------------


def runtime_dirs() -> list[Path]:
    # The helper script keeps pid files under the checkout's data/ even when
    # ENTERTAINER_DATA_DIR points elsewhere, so both places are checked.
    return list(dict.fromkeys([PATHS.root / "runtime", REPO_ROOT / "data" / "runtime"]))


def _alive(pid_file: Path) -> int | None:
    try:
        pid = int(pid_file.read_text().strip())
    except (OSError, ValueError):
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        return pid
    return pid


def require_app_stopped() -> None:
    """Refuse while the helper's app or studio holds the catalogue open."""
    for directory in runtime_dirs():
        for name in ("app", "studio"):
            pid = _alive(directory / f"{name}.pid")
            if pid:
                raise ReleaseError(
                    f"the managed {name} is running (pid {pid}) and has the catalogue open. "
                    "Run `./scripts/entertainer stop` first, or use `./scripts/entertainer pull`, "
                    "which stops and restarts it for you"
                )
