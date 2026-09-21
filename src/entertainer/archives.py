"""Safe extraction for archives obtained from another machine or the network."""

from __future__ import annotations

import zipfile
from pathlib import Path


def extract_zip(zf: zipfile.ZipFile, destination: Path) -> None:
    """Extract only members that remain under ``destination``.

    ``ZipFile.extractall`` accepts ``../`` paths and absolute names. Bundles
    are importable files, so their filenames are untrusted even on a local app.
    """
    destination = destination.resolve()
    for member in zf.infolist():
        target = (destination / member.filename).resolve()
        if target != destination and destination not in target.parents:
            raise ValueError(f"archive member escapes destination: {member.filename!r}")
    zf.extractall(destination)
