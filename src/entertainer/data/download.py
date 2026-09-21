"""Resumable downloads for the bulk datasets.

Every source here is a large static file, so downloads are resumed with HTTP
Range requests and skipped entirely when the local copy already matches the
server's content-length.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import httpx
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from ..archives import extract_zip
from ..config import IMDB_BASE, IMDB_FILES, MOVIELENS_URL, PATHS

_CHUNK = 1 << 20


def _remote_size(client: httpx.Client, url: str) -> int | None:
    try:
        r = client.head(url, follow_redirects=True, timeout=30)
        length = r.headers.get("content-length")
        return int(length) if length else None
    except httpx.HTTPError:
        return None


def fetch(url: str, dest: Path, progress: Progress | None = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True) as client:
        total = _remote_size(client, url)
        have = dest.stat().st_size if dest.exists() else 0
        if total is not None and have == total:
            return dest
        headers = {}
        mode = "wb"
        if have and total is not None and have < total:
            headers["Range"] = f"bytes={have}-"
            mode = "ab"
        else:
            have = 0

        task = None
        if progress is not None:
            task = progress.add_task(dest.name, total=total, completed=have)
        with client.stream("GET", url, headers=headers, timeout=None) as r:
            r.raise_for_status()
            with open(dest, mode) as fh:
                for chunk in r.iter_bytes(_CHUNK):
                    fh.write(chunk)
                    if progress is not None and task is not None:
                        progress.advance(task, len(chunk))
    return dest


def _progress() -> Progress:
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    )


def fetch_imdb(files: tuple[str, ...] = IMDB_FILES) -> list[Path]:
    out = []
    with _progress() as bar:
        for name in files:
            out.append(fetch(f"{IMDB_BASE}/{name}", PATHS.raw / "imdb" / name, bar))
    return out


def fetch_movielens() -> Path:
    """Download and unpack MovieLens-32M; returns the extracted directory."""
    target = PATHS.raw / "ml-32m"
    if (target / "ratings.csv").exists():
        return target
    archive = PATHS.raw / "ml-32m.zip"
    with _progress() as bar:
        fetch(MOVIELENS_URL, archive, bar)
    with zipfile.ZipFile(archive) as zf:
        extract_zip(zf, PATHS.raw)
    # The archive unpacks into ml-32m/ already; guard against a nested layout.
    if not (target / "ratings.csv").exists():
        for cand in PATHS.raw.glob("ml-32m*/ratings.csv"):
            return cand.parent
    return target
