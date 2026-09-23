"""Immutable provenance records for builds and evaluation reports."""

from __future__ import annotations

import hashlib
import json
import platform
import time
import uuid
from pathlib import Path
from typing import Any

from .config import PATHS


def _digest(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Write a never-overwritten JSON record and return its metadata."""
    # Deferred: fusion pulls in sklearn, which a manifest write has no other use for.
    from .models.fusion import fused_path

    PATHS.ensure()
    out = PATHS.reports / "manifests"
    out.mkdir(parents=True, exist_ok=True)
    manifest_id = f"{kind}-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    record = {
        "id": manifest_id,
        "kind": kind,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "artifacts": {
            "catalogue": _digest(PATHS.catalog_db),
            "fused": _digest(fused_path()),
            "prior": _digest(PATHS.artifacts / "population_prior.npz"),
        },
        **payload,
    }
    path = out / f"{manifest_id}.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    record["path"] = str(path)
    return record
