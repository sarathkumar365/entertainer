"""Search every source, try the best match, keep it only if it verifies.

The order is deliberate: rank everything first, then download one at a time
from the top, and verify each download before it reaches the library. A file
whose duration disagrees with the film's runtime is deleted and the next
candidate is tried — the name said it was the film, the file says otherwise,
and the file wins.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path

import httpx

from ..config import PATHS
from ..errors import NotAcquired
from .match import Candidate, Target, score

GB = 1 << 30


def library_dir() -> Path:
    raw = os.environ.get("ENTERTAINER_LIBRARY_DIR")
    return Path(raw).expanduser() if raw else PATHS.root / "library" / "Movies"


@dataclass
class Acquired:
    path: Path
    candidate: Candidate
    duration_min: float | None


async def rank(target: Target, sources: list) -> list[Candidate]:
    found: list[Candidate] = []
    async with AsyncExitStack() as stack:
        for src in sources:
            s = await stack.enter_async_context(src)
            found.extend(score(target, c) for c in await s.search(target))
    return sorted(found, key=lambda c: -c.score)


async def acquire(
    target: Target,
    sources: list,
    library: Path | None = None,
    *,
    reserve_gb: float = 5.0,
    max_tries: int = 3,
    max_gb: float | None = None,
    progress: Callable[[int, int], None] | None = None,
    log: Callable[[str], None] = print,
) -> Acquired:
    library = library or library_dir()
    staging = library.parent / ".staging"
    staging.mkdir(parents=True, exist_ok=True)

    async with AsyncExitStack() as stack:
        live = [await stack.enter_async_context(s) for s in sources]
        found: list[tuple[Candidate, object]] = []
        for s in live:
            for c in await s.search(target):
                found.append((score(target, c), s))
        found.sort(key=lambda p: -p[0].score)
        usable = [(c, s) for c, s in found if c.score > 0
                  and not (max_gb and c.size and c.size > max_gb * GB)]
        log(f"{len(found)} files seen, {len(usable)} match {target.folder}")
        if not usable:
            refused = "; ".join(sorted({c.why for c, _ in found})[:5]) or "nothing found"
            raise NotAcquired(f"No source had {target.folder}. Refused: {refused}")

        for cand, src in usable[:max_tries]:
            _check_disk(staging, cand.size, reserve_gb)
            part = staging / f"{target.folder}.{cand.ext or 'mkv'}.part"
            log(f"trying [{cand.source}] {cand.label} ({_gb(cand.size)}, score {cand.score:.1f})")
            try:
                await src.fetch(cand, part, progress)
            except Exception as e:  # noqa: BLE001 — any source failure means try the next
                log(f"  download failed: {e}")
                part.unlink(missing_ok=True)
                continue
            ok, dur, why = verify(part, target.runtime_min)
            if not ok:
                log(f"  rejected after download: {why}")
                part.unlink(missing_ok=True)
                continue
            final = library / target.folder / f"{target.folder}.{cand.ext or 'mkv'}"
            final.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(part, final)
            _refresh_jellyfin(log)
            return Acquired(final, cand, dur)
    raise NotAcquired(f"Every candidate for {target.folder} failed to download or verify.")


def verify(path: Path, runtime_min: int | None) -> tuple[bool, float | None, str]:
    """Playable, and about as long as the film is supposed to be."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=60, check=True,
        ).stdout
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        return False, None, f"ffprobe could not read it ({e})"
    info = json.loads(out)
    if not any(s.get("codec_type") == "video" for s in info.get("streams", [])):
        return False, None, "no video stream"
    dur = float(info.get("format", {}).get("duration") or 0) / 60
    if runtime_min:
        # Generous on purpose: cuts, frame rates and silent-era projection
        # speeds all move runtime. This catches samples and wrong films, not
        # director's cuts.
        tol = max(8.0, 0.2 * runtime_min)
        if abs(dur - runtime_min) > tol:
            return False, dur, f"{dur:.0f} min, expected about {runtime_min}"
    return True, dur, "ok"


def _check_disk(where: Path, size: int | None, reserve_gb: float) -> None:
    free = shutil.disk_usage(where).free
    if size and free - size < reserve_gb * GB:
        raise NotAcquired(
            f"Not enough disk: {_gb(size)} needed, {_gb(free)} free, {reserve_gb:g} GB kept in "
            "reserve. Free space or lower the reserve."
        )


def _refresh_jellyfin(log: Callable[[str], None]) -> None:
    url, key = os.environ.get("JELLYFIN_URL"), os.environ.get("JELLYFIN_API_KEY")
    if not (url and key):
        return
    try:
        httpx.post(f"{url.rstrip('/')}/Library/Refresh",
                   headers={"X-Emby-Token": key}, timeout=10).raise_for_status()
        log("Jellyfin library refresh requested")
    except httpx.HTTPError as e:
        log(f"Jellyfin refresh failed ({e}); the file is in place and a scan will find it")


def _gb(n: int | None) -> str:
    return f"{n / GB:.2f} GB" if n else "? GB"
