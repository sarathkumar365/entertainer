"""Is this machine set up to build, publish and pull — and is it doing so?

Everything here only looks. Each check reports a status, what it saw, and,
when something is missing, the command that fixes it, so Build Studio and
`ent release status` can say what to do next instead of just what is wrong.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import releases, resources
from .config import PATHS, has_tmdb

OK, WARN, MISSING, UNKNOWN = "ok", "warn", "missing", "unknown"

#: CUDA 12.8 wheels, which pyproject pins on Linux, need at least this driver.
MIN_DRIVER = 570
SUBPROCESS_TIMEOUT = 10
REPO_ROOT = Path(__file__).resolve().parents[2]
_BUILD_IN_NOTES = re.compile(r"build (cat-[0-9a-f]+)")
ROLE_ENV = "ENTERTAINER_ROLE"
BUILDER, PULLER = "builder", "puller"


def role() -> str:
    """A builder runs the full build and publishes; a puller only pulls.

    Cron jobs, CUDA drivers and 40 GiB of downloads matter on the first and
    are noise on the second, so every check is judged against the role.
    Without an override, a machine with a CUDA GPU is taken to be the builder.
    """
    override = os.environ.get(ROLE_ENV, "").strip().lower()
    if override in (BUILDER, PULLER):
        return override
    try:
        from .models import encoder

        return BUILDER if encoder._device() == "cuda" else PULLER
    except (ImportError, RuntimeError):
        return PULLER


@dataclass
class Check:
    id: str
    section: str
    label: str
    status: str
    detail: str = ""
    fix: str = ""
    data: dict = field(default_factory=dict)


def _run(*cmd: str) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(
            list(cmd), capture_output=True, text=True, check=False, timeout=SUBPROCESS_TIMEOUT
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


# --- this machine ----------------------------------------------------------


def _device(machine: str) -> Check:
    try:
        import torch  # noqa: F401
    except ImportError:
        if machine == PULLER:
            return Check("device", "machine", "Encoder hardware", OK,
                         "not needed — this machine pulls builds instead of making them")
        return Check(
            "device", "machine", "Encoder hardware", MISSING,
            "torch is not installed, so this machine cannot run the encoding stage",
            'uv pip install -e ".[encode,web,gpu]"   # on the GPU server',
        )
    from .models import encoder

    device = encoder._device()
    if device != "cuda":
        detail = {"mps": "Apple GPU (mps)", "cpu": "CPU only"}.get(device, device)
        slow = device == "cpu" and machine == BUILDER
        return Check(
            "device", "machine", "Encoder hardware", WARN if slow else OK,
            f"{detail} — fine for pulling, slow for a full build", data={"device": device},
        )
    smi = _run("nvidia-smi", "--query-gpu=name,driver_version,memory.total",
               "--format=csv,noheader,nounits")
    if smi is None or smi.returncode != 0 or not smi.stdout.strip():
        return Check("device", "machine", "Encoder hardware", OK, "CUDA GPU",
                     data={"device": device})
    # rsplit: a GPU name may contain commas. memory.total can be "[N/A]" on
    # vGPU and MIG setups.
    parts = [p.strip() for p in smi.stdout.splitlines()[0].rsplit(",", 2)]
    if len(parts) != 3:
        return Check("device", "machine", "Encoder hardware", OK, "CUDA GPU",
                     data={"device": device})
    name, driver, mem = parts
    size = f"{int(mem) / 1024:.0f} GiB, " if mem.isdigit() else ""
    detail = f"{name}, {size}driver {driver}"
    major = int(driver.split(".")[0]) if driver.split(".")[0].isdigit() else 0
    if major < MIN_DRIVER:
        return Check(
            "device", "machine", "Encoder hardware", WARN,
            f"{detail} — CUDA 12.8 torch needs driver {MIN_DRIVER}+",
            "upgrade the NVIDIA driver (e.g. sudo ubuntu-drivers install)",
            {"device": device, "driver": driver},
        )
    return Check("device", "machine", "Encoder hardware", OK, detail,
                 data={"device": device, "driver": driver, "gpu": name})


def _cf_gpu() -> Check:
    try:
        from implicit.gpu import HAS_CUDA
    except ImportError:
        HAS_CUDA = False
    if HAS_CUDA:
        return Check("cf_gpu", "machine", "Factorisation hardware", OK, "implicit with CUDA")
    return Check(
        "cf_gpu", "machine", "Factorisation hardware", OK,
        "CPU — takes minutes, so a GPU build of implicit is optional",
    )


def _memory() -> Check:
    try:
        b = resources.budget()
    except ValueError as exc:
        return Check("memory", "machine", "Memory budget", WARN, str(exc),
                     f"unset {resources.MEMORY_FRACTION_ENV} or give it a value in (0, 1]")
    return Check("memory", "machine", "Memory budget", OK, b.describe(),
                 data={"bytes": b.bytes, "total": b.total})


def _disk(machine: str) -> Check:
    PATHS.ensure()
    free = shutil.disk_usage(PATHS.root).free
    # A pulled build is well under 1 GiB; the raw downloads are what need room.
    status = OK if machine == PULLER or free >= 40 * 1024**3 else WARN
    return Check(
        "disk", "machine", "Disk", status, f"{free / 1024**3:.0f} GiB free at {PATHS.root}",
        "" if status == OK else "a full build wants about 40 GiB for the raw downloads",
    )


def _tmdb() -> Check:
    if has_tmdb():
        return Check("tmdb", "machine", "TMDB key", OK, "set in .env")
    return Check("tmdb", "machine", "TMDB key", MISSING, "needed to build or add titles",
                 "cp .env.example .env   # then paste TMDB_API_KEY")


# --- build -------------------------------------------------------------------


def _artefacts() -> Check:
    from .models import fusion

    have = {
        "catalogue": PATHS.catalog_db.exists(),
        "item space": fusion.exists(),
    }
    missing = [name for name, present in have.items() if not present]
    if not missing:
        return Check("artefacts", "build", "Model files", OK, "catalogue, item space and prior")
    return Check(
        "artefacts", "build", "Model files", MISSING, "missing: " + ", ".join(missing),
        "./scripts/entertainer build   # or, on a machine that does not build: "
        "./scripts/entertainer pull",
    )


def _last_build() -> Check:
    from .build_events import list_builds

    builds = list_builds()
    if not builds:
        return Check("last_build", "build", "Last build", UNKNOWN, "no build recorded here")
    b = builds[0]
    status = {"complete": OK, "running": OK, "failed": WARN, "interrupted": WARN}.get(
        b.get("status"), UNKNOWN
    )
    return Check(
        "last_build", "build", "Last build", status,
        f"{b.get('status')} · started {b.get('started_at')}",
        data={"id": b.get("id"), "status": b.get("status")},
    )


# --- releases ----------------------------------------------------------------


def _gh(repo: str, machine: str = PULLER) -> list[Check]:
    if shutil.which("gh") is None:
        return [Check("gh", "releases", "GitHub CLI", MISSING, "`gh` is not installed",
                      "brew install gh   # or see https://cli.github.com")]
    auth = _run("gh", "auth", "status", "--hostname", "github.com")
    if auth is None or auth.returncode != 0:
        return [Check("gh", "releases", "GitHub CLI", MISSING, "installed but not logged in",
                      "gh auth login")]
    checks = [Check("gh", "releases", "GitHub CLI", OK, "installed and logged in")]

    view = _run("gh", "repo", "view", repo, "--json", "visibility,url")
    if view is None or view.returncode != 0:
        checks.append(Check(
            "repo", "releases", "Builds repository", MISSING, f"{repo} not found or no access",
            f"gh repo create {repo} --private",
        ))
        return checks
    info = json.loads(view.stdout)
    if info.get("visibility") != "PRIVATE":
        checks.append(Check(
            "repo", "releases", "Builds repository", WARN,
            f"{repo} is {str(info.get('visibility')).lower()} — publishing is refused",
            f"gh repo edit {repo} --visibility private --accept-visibility-change-consequences",
        ))
        return checks
    checks.append(Check("repo", "releases", "Builds repository", OK, f"{repo} (private)",
                        data={"url": info.get("url")}))

    # The same endpoint `ent pull` resolves, which skips drafts (a publish
    # still uploading) and prereleases; a 404 means nothing is published.
    found = _run("gh", "api", f"repos/{repo}/releases/latest")
    latest = json.loads(found.stdout) if found and found.returncode == 0 and found.stdout.strip() \
        else None
    if latest is None:
        checks.append(Check(
            "latest", "releases", "Latest release", MISSING, "nothing published yet",
            "ent setup && ent release publish   # on the build server",
        ))
        return checks
    size = sum(a.get("size", 0) for a in latest.get("assets", []))
    match = _BUILD_IN_NOTES.search(latest.get("body") or "")
    build_id = match.group(1) if match else None
    checks.append(Check(
        "latest", "releases", "Latest release", OK,
        f"{latest.get('tag_name')} · {latest.get('published_at')} · {size / 1e6:.0f} MB",
        data={"tag": latest.get("tag_name"), "build_id": build_id,
              "published_at": latest.get("published_at"), "url": latest.get("html_url")},
    ))
    checks.append(_this_machine(build_id, machine))
    return checks


def _this_machine(published: str | None, machine: str) -> Check:
    import duckdb

    from . import store
    from .bundle import BUILD_META

    # A read-only connect creates an empty catalogue when there is none,
    # which would then pass for a built one on every later check.
    if not PATHS.catalog_db.exists():
        return Check("pulled", "releases", "This machine", MISSING, "no catalogue yet",
                     "./scripts/entertainer pull")
    try:
        with store.session(read_only=True) as con:
            local = store.get_meta(con, BUILD_META)
    except duckdb.IOException:
        return Check("pulled", "releases", "This machine", UNKNOWN,
                     "a build has the catalogue open; check again when it finishes")
    if local and published and local == published:
        where = "published from here" if machine == BUILDER else "pulled"
        return Check("pulled", "releases", "This machine", OK,
                     f"has the latest build ({local}, {where})")
    if machine == BUILDER:
        return Check("pulled", "releases", "This machine", WARN,
                     "the catalogue here has not been published yet",
                     "ent release publish")
    if local:
        return Check("pulled", "releases", "This machine", WARN,
                     f"has {local}; the latest is {published or 'unknown'}",
                     "./scripts/entertainer pull")
    return Check(
        "pulled", "releases", "This machine", WARN,
        "catalogue was built here, not pulled",
        "./scripts/entertainer pull   # if this machine should not build",
    )


# --- schedule ----------------------------------------------------------------


def suggested_cron() -> str:
    ent = REPO_ROOT / ".venv" / "bin" / "ent"
    logs = PATHS.root / "runtime" / "logs"
    # The directory may not exist, and a failed redirect would silently skip
    # the whole job; the group sends both commands' output to the log.
    return (
        f"0 3 * * 0  mkdir -p {logs} && cd {REPO_ROOT} && "
        f"{{ {ent} setup && {ent} release publish; }} >> {logs}/scheduled.log 2>&1"
    )


def _cron(machine: str) -> Check:
    listed = _run("crontab", "-l")
    lines = []
    if listed is not None and listed.returncode == 0:
        lines = [ln.strip() for ln in listed.stdout.splitlines()
                 if ln.strip() and not ln.lstrip().startswith("#")]
    publishing = [ln for ln in lines if "release publish" in ln]
    if publishing:
        schedule = " ".join(publishing[0].split()[:5])
        return Check("cron", "schedule", "Scheduled rebuild", OK,
                     f"cron `{schedule}`", data={"line": publishing[0], "schedule": schedule})
    building = [ln for ln in lines if re.search(r"\bent\b.*\bsetup\b", ln)]
    if building:
        return Check(
            "cron", "schedule", "Scheduled rebuild", WARN,
            "a scheduled build exists but never publishes", "append: && ent release publish",
            {"line": building[0]},
        )
    if machine == PULLER:
        return Check("cron", "schedule", "Scheduled rebuild", OK,
                     "not needed here — the build server publishes, this machine pulls")
    return Check(
        "cron", "schedule", "Scheduled rebuild", MISSING,
        "no cron job builds and publishes",
        f"(crontab -l; echo '{suggested_cron()}') | crontab -",
    )


# --- all of it ---------------------------------------------------------------


def collect(repo: str | None = None) -> dict:
    repo = releases.releases_repo(repo)
    machine = role()
    checks: list[Check] = [
        _device(machine), _cf_gpu(), _memory(), _disk(machine), _tmdb(),
        _artefacts(), _last_build(),
        *_gh(repo, machine),
        _cron(machine),
    ]
    worst = next((s for s in (MISSING, WARN, UNKNOWN) if any(c.status == s for c in checks)), OK)
    return {
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo": repo,
        "role": machine,
        "overall": worst,
        "checks": [asdict(c) for c in checks],
    }
