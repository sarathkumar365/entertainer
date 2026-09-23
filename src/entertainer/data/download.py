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


def _remote_head(client: httpx.Client, url: str) -> tuple[int | None, str | None]:
    """Content-length and a version validator (ETag, else Last-Modified)."""
    try:
        r = client.head(url, follow_redirects=True, timeout=30)
    except httpx.HTTPError:
        return None, None
    length = r.headers.get("content-length")
    validator = r.headers.get("etag") or r.headers.get("last-modified")
    return (int(length) if length else None), validator


def _validator_path(dest: Path) -> Path:
    return dest.with_name(dest.name + ".validator")


def fetch(url: str, dest: Path, progress: Progress | None = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    stamp = _validator_path(dest)
    with httpx.Client(follow_redirects=True) as client:
        total, validator = _remote_head(client, url)
        have = dest.stat().st_size if dest.exists() else 0
        saved = stamp.read_text().strip() if stamp.exists() else None
        # IMDb republishes its dumps daily. A size match or a byte-range resume
        # is only meaningful against the same version of the file: resuming a
        # yesterday's partial copy against today's file splices two different
        # gzip streams into one corrupt file.
        same_version = validator is None or saved is None or saved == validator
        if total is not None and have == total and same_version:
            return dest
        headers = {}
        resuming = bool(
            have and total is not None and have < total and validator and saved == validator
        )
        if resuming:
            headers["Range"] = f"bytes={have}-"
            headers["If-Range"] = validator
        else:
            have = 0
        if validator:
            stamp.write_text(validator)

        task = None
        if progress is not None:
            task = progress.add_task(dest.name, total=total, completed=have)
        with client.stream("GET", url, headers=headers, timeout=None) as r:
            r.raise_for_status()
            # If-Range: the server answers 200 with the whole file when the
            # version changed, and 206 only when the tail is safe to append.
            mode = "ab" if resuming and r.status_code == 206 else "wb"
            if mode == "wb" and have and progress is not None and task is not None:
                progress.reset(task, total=total)
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
