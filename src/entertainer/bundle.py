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

import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import duckdb

from .config import PATHS
from .store import connect

MANIFEST = "bundle.json"
FORMAT_VERSION = 1

# Everything the interface and the recommender need, by role.
CORE = ("titles.parquet",)
SPACE = ("fused.npz", "population_prior.npz")
ENCODE = ("content.npy", "content_ids.npy")


@dataclass
class BundleInfo:
    path: Path
    titles: int
    contents: list[str]
    bytes: int


def _copy_if_present(src: Path, arc: str, zf: zipfile.ZipFile, out: list[str]) -> None:
    if src.exists():
        zf.write(src, arc)
        out.append(arc)


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
        f"COPY (SELECT * FROM live.titles) TO '{staging / 'titles.parquet'}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    scratch.close()

    contents: list[str] = []
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.write(staging / "titles.parquet", "titles.parquet")
        contents.append("titles.parquet")
        if include_space:
            for name in SPACE:
                for base in (PATHS.embeddings, PATHS.artifacts):
                    _copy_if_present(base / name, name, zf, contents)
        if include_encodings:
            for name in ENCODE:
                _copy_if_present(PATHS.embeddings / name, name, zf, contents)
        zf.writestr(
            MANIFEST,
            json.dumps(
                {
                    "format": FORMAT_VERSION,
                    "titles": int(n_titles),
                    "contents": contents,
                    "note": "catalogue only — verdicts travel via `ent export`",
                },
                indent=2,
            ),
        )
    shutil.rmtree(staging)
    tmp.replace(path)
    return BundleInfo(
        path=path, titles=int(n_titles), contents=contents, bytes=path.stat().st_size
    )


def inspect(path: Path) -> dict:
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read(MANIFEST))


def restore(path: Path, overwrite: bool = False) -> dict:
    """Unpack a bundle into this machine's data directory.

    Refuses to clobber an existing catalogue unless told to, because the
    catalogue on this machine may carry verdicts that the bundle does not.
    """
    path = Path(path)
    manifest = inspect(path)
    PATHS.ensure()

    existing_events = 0
    if PATHS.catalog_db.exists():
        con = connect(read_only=True)
        try:
            existing_events = con.execute("SELECT count(*) FROM events").fetchone()[0]
        except duckdb.Error:
            existing_events = 0
        con.close()
        if existing_events and not overwrite:
            raise RuntimeError(
                f"this machine already has {existing_events} recorded events; "
                "run `ent export` to save them first, then pass --overwrite"
            )

    staging = PATHS.root / ".bundle-restore"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    with zipfile.ZipFile(path) as zf:
        zf.extractall(staging)

    # Rebuild the catalogue table through the normal schema so that any
    # columns added since the bundle was written exist and are simply null,
    # rather than the restore failing on a shape mismatch.
    con = connect()
    columns = [r[1] for r in con.execute("PRAGMA table_info('titles')").fetchall()]
    parquet = staging / "titles.parquet"
    available = {
        r[0]
        for r in con.execute(
            f"SELECT column_name FROM (DESCRIBE SELECT * FROM '{parquet}')"
        ).fetchall()
    }
    projection = ", ".join(
        c if c in available else f"NULL AS {c}" for c in columns
    )
    con.execute("DELETE FROM titles")
    con.execute(f"INSERT INTO titles ({', '.join(columns)}) SELECT {projection} FROM '{parquet}'")
    restored = con.execute("SELECT count(*) FROM titles").fetchone()[0]
    con.close()

    for name in SPACE + ENCODE:
        src = staging / name
        if not src.exists():
            continue
        dest = PATHS.artifacts if name.endswith("prior.npz") else PATHS.embeddings
        shutil.copy2(src, dest / name)

    shutil.rmtree(staging)
    manifest["restored_titles"] = int(restored)
    manifest["preserved_events"] = int(existing_events)
    return manifest
