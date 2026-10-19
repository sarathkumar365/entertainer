"""Putting a verified file where Jellyfin expects it, and telling Jellyfin.

The layout is Jellyfin's: ``Movies/<Title> (<Year>)/<Title> (<Year>).<ext>``.
Every part of that name comes from TMDB, never from the file or its sender:
a label from a Telegram bot is untrusted text and must not become a path.
"""

from __future__ import annotations

import errno
import os
import shutil
from collections.abc import Callable
from pathlib import Path

import httpx

from ..acquire import Fetched, Target
from ..acquire.match import VIDEO_EXTS
from ..acquire.verify import container
from ..config import AcquireSettings
from ..errors import NotAcquired


def existing(target: Target, library: Path) -> Path | None:
    folder = library / target.folder
    if not folder.is_dir():
        return None
    return next(
        (p for p in folder.iterdir() if p.suffix.lstrip(".").lower() in VIDEO_EXTS), None
    )


def file_into(fetched: Fetched, target: Target, library: Path) -> Path:
    ext = fetched.candidate.ext
    if ext not in VIDEO_EXTS:
        ext = container(fetched.path) or "mkv"
    dest = library / target.folder / f"{target.folder}.{ext}"
    if dest.exists():
        raise NotAcquired(f"{dest} already exists; the new copy was not filed over it.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(fetched.path, dest)
    except OSError as e:
        # Staging sits inside the library, so this is a rename; a library
        # spread over mounts is the one case where it becomes a copy.
        if e.errno != errno.EXDEV:
            raise
        shutil.move(fetched.path, dest)
    return dest


def refresh_jellyfin(settings: AcquireSettings, log: Callable[[str], None]) -> None:
    if not (settings.jellyfin_url and settings.jellyfin_key):
        return
    try:
        httpx.post(f"{settings.jellyfin_url.rstrip('/')}/Library/Refresh",
                   headers={"X-Emby-Token": settings.jellyfin_key}, timeout=10).raise_for_status()
        log("Jellyfin library refresh requested")
    except httpx.HTTPError as e:
        log(f"Jellyfin refresh failed ({e}); the file is in place and a scan will find it")
