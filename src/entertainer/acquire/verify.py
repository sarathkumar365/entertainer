"""Checking a downloaded file before it can reach the library.

The name said it was the film; the file gets the last word. ffprobe must find
a video stream, and the running time must be close to the film's. That catches
samples, trailers, wrong films and archives posing as video — not a right-length
wrong film, which only the user can catch ("wrong file").
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


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


# ffprobe's format names for the containers a film is likely to arrive in.
_CONTAINERS = {"matroska": "mkv", "mp4": "mp4", "mov": "mov", "avi": "avi", "mpegts": "ts",
               "webm": "webm", "mpeg": "mpg", "ogg": "ogv"}


def container(path: Path) -> str | None:
    """The extension the file's actual container calls for, when its name gave none."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=format_name", "-of", "json",
             str(path)],
            capture_output=True, text=True, timeout=60, check=True,
        ).stdout
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    names = str(json.loads(out).get("format", {}).get("format_name") or "").split(",")
    # ffprobe lists a family ("mov,mp4,m4a,..."); the table's order picks the usual one.
    return next((ext for n, ext in _CONTAINERS.items() if n in names), None)
