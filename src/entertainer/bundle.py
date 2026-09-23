"""Move a built catalogue between machines.

Building the catalogue takes about four hours, almost all of it TMDB round
trips, and the result is derived data that git should not carry. But a person
with two computers should not have to do it twice, and the rating interface
is far more useful on whichever machine they are actually sitting at.

A bundle is the built state without the raw inputs: the catalogue table as
compressed Parquet, and optionally the item space needed for
recommendations. The live DuckDB file is around 300MB, nearly all of it
write-amplification from the build; the same rows as zstd Parquet are about
21MB.

Verdicts deliberately do not travel in a bundle. They are tiny, they are the
one irreplaceable thing here, and silently merging two divergent logs is a
good way to lose some — `ent export` and `ent import` handle them explicitly.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import zipfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np

from . import store
from .archives import extract_zip
from .config import ENCODER_MODEL, PATHS
from .errors import EntertainerError
from .manifests import _digest
from .pipeline import PUBLISHED_BUILD_KEY
from .store import connect

MANIFEST = "bundle.json"
FORMAT_VERSION = 2
READABLE_FORMATS = (1, 2)

# Everything the interface and the recommender need, by role.
CORE = ("titles.parquet",)
SPACE = ("fused.npz", "population_prior.npz")
ENCODE = ("content.npy", "content_ids.npy")

#: Meta key naming the build the local catalogue came from, so `ent pull`
#: can tell when there is nothing new.
BUILD_META = PUBLISHED_BUILD_KEY


class BundleError(EntertainerError):
    """A bundle this code cannot safely import."""


@dataclass
class BundleInfo:
    path: Path
    titles: int
    contents: list[str]
    bytes: int
    manifest: dict


def _latest_build_id() -> str | None:
    # Manifest ids start with the kind and a sortable timestamp.
    found = sorted((PATHS.reports / "manifests").glob("build-*.json"))
    for path in reversed(found):
        with suppress(OSError, ValueError, KeyError):
            return str(json.loads(path.read_text(encoding="utf-8"))["id"])
    return None


def _content_id(sha256: dict[str, str]) -> str:
    """Name a bundle by what is in it, not by which build last ran here.

    `ent pull` treats an equal id as "already up to date". A label taken from
    the newest build manifest stays the same after a partial rebuild (`data
    embed`, `data fuse`) and travels with a machine that pulled someone
    else's catalogue, so two different catalogues could share it.
    """
    joined = "\n".join(f"{name}={digest}" for name, digest in sorted(sha256.items()))
    return "cat-" + hashlib.sha256(joined.encode()).hexdigest()[:16]


def _source(name: str) -> Path | None:
    bases = (PATHS.embeddings, PATHS.artifacts) if name in SPACE else (PATHS.embeddings,)
    for base in bases:
        if (base / name).exists():
            return base / name
    return None


def export(
    path: Path,
    include_space: bool = True,
    include_encodings: bool = False,
) -> BundleInfo:
    """Write a bundle.

    ``include_space`` carries the fused item space and population prior, which
    `ent recs`, `ent taste` and `ent audit` need. Without it the bundle still
    supports rating and search, which is the common case for a second machine.

    ``include_encodings`` adds the raw content embeddings. Only `ent add`
    needs them, and only to place a brand-new title into the space, so they
    are off by default — they are the largest single file and the least used.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.building"

    con = connect(read_only=True)
    n_titles = con.execute("SELECT count(*) FROM titles").fetchone()[0]
    con.close()

    staging = path.parent / f".{path.name}.stage"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    # Parquet rather than a copy of the database file: the live file carries
    # roughly ten times its own content in accumulated write amplification.
    scratch = duckdb.connect()
    scratch.execute(f"ATTACH '{PATHS.catalog_db}' AS live (READ_ONLY)")
    scratch.execute(
        # Ordered, so the same catalogue always writes the same bytes and so
        # the same content id.
        f"COPY (SELECT * FROM live.titles ORDER BY item_id) TO '{staging / 'titles.parquet'}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    columns = [r[1] for r in scratch.execute("PRAGMA table_info('live.titles')").fetchall()]
    scratch.close()

    files: dict[str, Path] = {"titles.parquet": staging / "titles.parquet"}
    wanted = (SPACE if include_space else ()) + (ENCODE if include_encodings else ())
    for name in wanted:
        src = _source(name)
        if src is not None:
            files[name] = src

    digests = {name: _digest(src) for name, src in files.items()}
    manifest: dict = {
        "format": FORMAT_VERSION,
        "build_id": _content_id(digests),
        "source_build": _latest_build_id(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "schema_version": store.SCHEMA_VERSION,
        "titles": int(n_titles),
        "titles_columns": columns,
        "contents": list(files),
        "sha256": digests,
        "note": "catalogue only — verdicts travel via `ent export`",
    }
    if "content.npy" in files:
        dim = int(np.load(files["content.npy"], mmap_mode="r").shape[1])
        # The configured model, not a record of the encoding run: the build
        # does not persist which model produced content.npy.
        manifest["encoder"] = {"model": ENCODER_MODEL, "dim": dim}

    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for name, src in files.items():
            zf.write(src, name)
        zf.writestr(MANIFEST, json.dumps(manifest, indent=2))
    shutil.rmtree(staging)
    tmp.replace(path)
    return BundleInfo(
        path=path,
        titles=int(n_titles),
        contents=list(files),
        bytes=path.stat().st_size,
        manifest=manifest,
    )


def inspect(path: Path) -> dict:
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read(MANIFEST))


def check_compatible(manifest: dict) -> list[str]:
    """Refuse a bundle this code cannot read; return warnings for one it can.

    Cheap enough to run on a manifest downloaded on its own, before fetching
    the archive it describes.
    """
    fmt = manifest.get("format")
    if fmt not in READABLE_FORMATS:
        raise BundleError(
            f"bundle format {fmt!r} is not one this version reads "
            f"({', '.join(map(str, READABLE_FORMATS))}); update this checkout with `git pull`"
        )
    if fmt == 1:
        return ["format-1 bundle: it carries no checksums, so its files cannot be verified"]

    theirs = manifest.get("schema_version")
    if not isinstance(theirs, int) or theirs > store.SCHEMA_VERSION:
        raise BundleError(
            f"bundle was written at catalogue schema {theirs!r} but this code reads up to "
            f"{store.SCHEMA_VERSION}; update this checkout with `git pull` first"
        )
    unknown = sorted(set(manifest.get("titles_columns") or []) - set(store.titles_columns()))
    if unknown:
        raise BundleError(
            f"bundle catalogue has columns this code does not know ({', '.join(unknown)}); "
            "update this checkout with `git pull` first"
        )
    hashes = manifest.get("sha256") or {}
    missing = [n for n in manifest.get("contents", []) if n not in hashes]
    if missing:
        raise BundleError(f"bundle manifest has no checksum for {', '.join(missing)}")
    return []


def verify(directory: Path, manifest: dict) -> None:
    """Every listed file present and byte-identical to what was exported."""
    if manifest.get("format") == 1:
        return
    for name, expected in (manifest.get("sha256") or {}).items():
        actual = _digest(directory / name)
        if actual is None:
            raise BundleError(f"bundle is missing {name}")
        if actual != expected:
            raise BundleError(
                f"{name} does not match its checksum; the bundle is corrupt or was altered"
            )


def _remap_json_ids(raw: str | None, remap: dict[int, int]) -> str:
    return json.dumps([remap[int(i)] for i in json.loads(raw or "[]") if int(i) in remap])


def restore(path: Path, overwrite: bool = False) -> dict:
    """Unpack a bundle into this machine's data directory.

    Refuses to clobber an existing catalogue unless told to, because the
    catalogue on this machine may carry verdicts that the bundle does not.
    With ``overwrite`` those verdicts are kept, moved onto the new catalogue's
    item ids by IMDb id.
    """
    path = Path(path)
    manifest = inspect(path)
    warnings = check_compatible(manifest)
    PATHS.ensure()

    staging = PATHS.root / ".bundle-restore"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    incoming: list[tuple[Path, Path]] = []
    try:
        with zipfile.ZipFile(path) as zf:
            extract_zip(zf, staging)
        # Before the database is opened: a torn download must not get as far
        # as deleting the catalogue it was meant to replace.
        verify(staging, manifest)

        parquet = staging / "titles.parquet"
        if not parquet.exists():
            raise BundleError("bundle is missing titles.parquet")

        existing_events = 0
        if PATHS.catalog_db.exists():
            con = connect(read_only=True)
            try:
                existing_events = con.execute("SELECT count(*) FROM events").fetchone()[0]
            except duckdb.Error:
                existing_events = 0
            con.close()
            if existing_events and not overwrite:
                raise BundleError(
                    f"this machine already has {existing_events} recorded events; "
                    "run `ent export` to save them first, then pass --overwrite"
                )

        # Copied beside the live files now so that, once the catalogue commits,
        # all that remains is a rename per file.
        for name in SPACE + ENCODE:
            src = staging / name
            if not src.exists():
                continue
            dest = (PATHS.artifacts if name.endswith("prior.npz") else PATHS.embeddings) / name
            tmp = dest.with_name(f".{name}.incoming")
            shutil.copy2(src, tmp)
            incoming.append((tmp, dest))

        restored = _replace_titles(parquet, manifest)

        for tmp, dest in incoming:
            os.replace(tmp, dest)
        incoming.clear()
    finally:
        for tmp, _ in incoming:
            tmp.unlink(missing_ok=True)
        shutil.rmtree(staging, ignore_errors=True)

    manifest["restored_titles"] = int(restored)
    manifest["preserved_events"] = int(existing_events)
    manifest["warnings"] = warnings
    return manifest


def _replace_titles(parquet: Path, manifest: dict) -> int:
    con = connect()
    try:
        # Rebuild the catalogue table through the normal schema so that any
        # columns added since the bundle was written exist and are simply
        # null, rather than the restore failing on a shape mismatch.
        columns = [r[1] for r in con.execute("PRAGMA table_info('titles')").fetchall()]
        available = {
            r[0]
            for r in con.execute(
                f"SELECT column_name FROM (DESCRIBE SELECT * FROM '{parquet}')"
            ).fetchall()
        }
        if "imdb_id" not in available:
            raise BundleError("bundle catalogue has no IMDb ids for safe profile preservation")
        unknown = sorted(available - set(columns))
        if unknown:
            raise BundleError(
                f"bundle catalogue has columns this code does not know ({', '.join(unknown)}); "
                "update this checkout with `git pull` first"
            )
        projection = ", ".join(c if c in available else f"NULL AS {c}" for c in columns)

        # Item ids are positional. Preserve an existing profile only when every
        # referenced title can be mapped through its stable IMDb id; leaving an
        # unmapped numeric id in place would silently attach a verdict to a
        # different title after the replacement.
        con.execute("CREATE OR REPLACE TEMP TABLE _bundle_titles AS SELECT * FROM read_parquet(?)", [str(parquet)])
        con.execute(
            "CREATE OR REPLACE TEMP TABLE _bundle_remap AS "
            "SELECT old.item_id AS old_id, incoming.item_id AS new_id "
            "FROM titles old JOIN _bundle_titles incoming USING (imdb_id)"
        )
        unmapped = 0
        for table in ("events", "impressions", "validation_cases"):
            unmapped += int(
                con.execute(
                    f"SELECT count(*) FROM {table} x LEFT JOIN _bundle_remap r "
                    "ON x.item_id = r.old_id WHERE r.new_id IS NULL"
                ).fetchone()[0]
            )
        if unmapped:
            raise BundleError(
                f"cannot preserve {unmapped} event(s): their titles are absent from this bundle; "
                "export the profile first, then import it after restoring"
            )

        # Id lists inside JSON cannot be joined, so they are remapped here.
        batches = con.execute(
            "SELECT batch_id, item_ids, full_ranking, ridge_ranking FROM validation_batches"
        ).fetchall()
        slate = store.last_slate(con)
        remap: dict[int, int] = {}
        if batches or slate:
            remap = {int(o): int(n) for o, n in con.execute("SELECT old_id, new_id FROM _bundle_remap").fetchall()}

        con.execute("BEGIN TRANSACTION")
        for table in ("events", "impressions", "validation_cases"):
            con.execute(
                f"UPDATE {table} SET item_id = r.new_id FROM _bundle_remap r "
                f"WHERE {table}.item_id = r.old_id"
            )
        for batch_id, item_ids, full, ridge in batches:
            con.execute(
                "UPDATE validation_batches SET item_ids = ?, full_ranking = ?, ridge_ranking = ? "
                "WHERE batch_id = ?",
                [_remap_json_ids(item_ids, remap), _remap_json_ids(full, remap),
                 _remap_json_ids(ridge, remap), batch_id],
            )
        if slate:
            store.set_last_slate(con, [remap[i] for i in slate if i in remap])
        con.execute("DELETE FROM titles")
        con.execute(
            f"INSERT INTO titles ({', '.join(columns)}) SELECT {projection} FROM _bundle_titles"
        )
        if manifest.get("build_id"):
            store.set_meta(con, BUILD_META, manifest["build_id"])
        else:
            # A format-1 bundle carries no id, so nothing can vouch for this
            # catalogue being any published build.
            con.execute("DELETE FROM meta WHERE key = ?", [BUILD_META])
        restored = con.execute("SELECT count(*) FROM titles").fetchone()[0]
        con.execute("COMMIT")
    except Exception:
        with suppress(duckdb.Error):
            con.execute("ROLLBACK")
        raise
    finally:
        con.close()
    return int(restored)
