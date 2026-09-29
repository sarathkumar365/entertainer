"""Round-tripping a built catalogue between machines.

The catalogue takes four hours to build, so this path exists to avoid doing
it twice. The risk it has to be safe against is clobbering verdicts — the one
thing on either machine that cannot be rebuilt.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest


def _fresh(tmp_path, monkeypatch, name):
    """A self-contained data directory with reloaded modules pointing at it."""
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("ENTERTAINER_DATA_DIR", str(root))
    from entertainer import bundle, config, store

    importlib.reload(config)
    importlib.reload(store)
    importlib.reload(bundle)
    config.PATHS.ensure()
    return config, store, bundle


def _populate(store, n=40):
    con = store.connect()
    for i in range(n):
        con.execute(
            """
            INSERT INTO titles (item_id, imdb_id, kind, title, original_title, year,
                language, runtime, genres, imdb_rating, imdb_votes, quality,
                overview, poster_path)
            VALUES (?, ?, 'movie', ?, ?, 2024, ?, 120, ['Drama'], 7.5, 1000, 0.6, ?, ?)
            """,
            [i, f"tt{i:07d}", f"Film {i}", f"Film {i}",
             ["ml", "ta", "en", "ko"][i % 4], f"Synopsis {i}", f"/p{i}.jpg"],
        )
    con.close()


def test_bundle_round_trips_the_catalogue(tmp_path, monkeypatch):
    config, store, bundle = _fresh(tmp_path, monkeypatch, "source")
    _populate(store)
    np.save(config.PATHS.embeddings / "content_ids.npy", np.arange(40, dtype=np.int32))
    np.savez(config.PATHS.embeddings / "fused.npz", item_ids=np.arange(40))

    out = tmp_path / "b.zip"
    info = bundle.export(out, include_space=True)
    assert info.titles == 40
    assert "titles.parquet" in info.contents
    assert "fused.npz" in info.contents
    assert out.stat().st_size > 0

    config2, store2, bundle2 = _fresh(tmp_path, monkeypatch, "target")
    manifest = bundle2.restore(out)
    assert manifest["restored_titles"] == 40

    con = store2.connect(read_only=True)
    titles = con.execute("SELECT count(*) FROM titles").fetchone()[0]
    sample = con.execute("SELECT title, language FROM titles WHERE item_id = 7").fetchone()
    con.close()
    assert titles == 40
    assert sample == ("Film 7", "ko")
    assert (config2.PATHS.embeddings / "fused.npz").exists()


def test_a_bundle_carrying_the_old_population_prior_still_restores(tmp_path, monkeypatch):
    """A build published before the population prior was cut must keep working.

    `build-20260925-1806` was already on the releases repository when the prior
    was removed, and its manifest lists `population_prior.npz` with a checksum.
    `verify` walks the *manifest's* checksum list rather than this code's SPACE
    tuple, so the file is still checked; the restore loop then simply does not
    copy it out. Neither half may raise, or pulling that release would fail on
    every machine.
    """
    config, store, bundle = _fresh(tmp_path, monkeypatch, "old-source")
    _populate(store)
    np.save(config.PATHS.embeddings / "content_ids.npy", np.arange(40, dtype=np.int32))
    np.savez(config.PATHS.embeddings / "fused.npz", item_ids=np.arange(40))

    out = tmp_path / "old.zip"
    bundle.export(out, include_space=True)

    # Rewrite the archive the way the older code would have written it: the
    # prior file present, listed in `contents`, and checksummed.
    prior = tmp_path / "population_prior.npz"
    np.savez(prior, mean=np.zeros(4))
    digest = bundle._digest(prior)

    import json
    import zipfile

    rebuilt = tmp_path / "old-with-prior.zip"
    with zipfile.ZipFile(out) as src, zipfile.ZipFile(rebuilt, "w") as dst:
        manifest = json.loads(src.read(bundle.MANIFEST))
        manifest["contents"].append("population_prior.npz")
        manifest["sha256"]["population_prior.npz"] = digest
        for item in src.infolist():
            if item.filename == bundle.MANIFEST:
                continue
            dst.writestr(item, src.read(item.filename))
        dst.writestr(bundle.MANIFEST, json.dumps(manifest))
        dst.write(prior, "population_prior.npz")

    config2, store2, bundle2 = _fresh(tmp_path, monkeypatch, "old-target")
    assert "population_prior.npz" in bundle2.inspect(rebuilt)["contents"]
    manifest = bundle2.restore(rebuilt)

    assert manifest["restored_titles"] == 40
    assert (config2.PATHS.embeddings / "fused.npz").exists()
    # Verified on the way in, then dropped: nothing reads it any more.
    assert not (config2.PATHS.artifacts / "population_prior.npz").exists()


def test_bundle_refuses_to_clobber_recorded_verdicts(tmp_path, monkeypatch):
    """Verdicts are the one thing that cannot be rebuilt."""
    config, store, bundle = _fresh(tmp_path, monkeypatch, "src2")
    _populate(store)
    out = tmp_path / "b2.zip"
    bundle.export(out, include_space=False)

    config2, store2, bundle2 = _fresh(tmp_path, monkeypatch, "tgt2")
    _populate(store2, n=5)
    con = store2.connect()
    # Same IMDb title, deliberately different positional id. A restore must
    # remap this event to the source bundle's id rather than preserve ``100``.
    con.execute("UPDATE titles SET item_id = 100 WHERE item_id = 1")
    store2.log_event(con, 100, "rate", 9.0, "manual", {"verdict": "love"})
    con.close()

    with pytest.raises(RuntimeError, match="already has 1 recorded events"):
        bundle2.restore(out)

    manifest = bundle2.restore(out, overwrite=True)
    assert manifest["preserved_events"] == 1

    con = store2.connect(read_only=True)
    # The catalogue was replaced; the verdict survived.
    assert con.execute("SELECT count(*) FROM titles").fetchone()[0] == 40
    assert con.execute("SELECT count(*) FROM events").fetchone()[0] == 1
    assert con.execute("SELECT item_id FROM events").fetchone()[0] == 1
    con.close()


def test_bundle_rejects_path_traversal_before_extracting(tmp_path, monkeypatch):
    import zipfile

    _config, _store, bundle = _fresh(tmp_path, monkeypatch, "unsafe")
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(bundle.MANIFEST, '{"format": 1, "contents": []}')
        zf.writestr("../outside", "nope")
    with pytest.raises(ValueError, match="escapes destination"):
        bundle.restore(archive)
    assert not (tmp_path / "outside").exists()


def test_space_can_be_left_out_for_a_rating_only_machine(tmp_path, monkeypatch):
    config, store, bundle = _fresh(tmp_path, monkeypatch, "src3")
    _populate(store)
    np.savez(config.PATHS.embeddings / "fused.npz", item_ids=np.arange(40))
    out = tmp_path / "b3.zip"
    info = bundle.export(out, include_space=False)
    assert info.contents == ["titles.parquet"]


def test_restore_tolerates_a_bundle_missing_newer_columns(tmp_path, monkeypatch):
    """A bundle written before a migration must still restore."""
    import duckdb

    config, store, bundle = _fresh(tmp_path, monkeypatch, "src4")
    _populate(store, n=6)
    out = tmp_path / "b4.zip"
    bundle.export(out, include_space=False)

    # Strip a column from the packed parquet, simulating an older bundle.
    import shutil
    import zipfile

    older = tmp_path / "older.zip"
    with zipfile.ZipFile(out) as src, zipfile.ZipFile(older, "w") as dst:
        for item in src.namelist():
            if item == "titles.parquet":
                tmp_pq = tmp_path / "t.parquet"
                with open(tmp_pq, "wb") as fh:
                    fh.write(src.read(item))
                con = duckdb.connect()
                con.execute(
                    f"COPY (SELECT * EXCLUDE (keywords_at) FROM '{tmp_pq}') "
                    f"TO '{tmp_path / 'trimmed.parquet'}' (FORMAT PARQUET)"
                )
                con.close()
                dst.write(tmp_path / "trimmed.parquet", "titles.parquet")
            elif item == bundle.MANIFEST:
                # Bundles that old predate checksums.
                dst.writestr(item, '{"format": 1, "titles": 6, "contents": ["titles.parquet"]}')
            else:
                dst.writestr(item, src.read(item))
    del shutil

    config5, store5, bundle5 = _fresh(tmp_path, monkeypatch, "tgt4")
    manifest = bundle5.restore(older)
    assert manifest["restored_titles"] == 6
    con = store5.connect(read_only=True)
    assert con.execute("SELECT keywords_at FROM titles LIMIT 1").fetchone()[0] is None
    con.close()


def test_inspect_reads_the_manifest_without_unpacking(tmp_path, monkeypatch):
    config, store, bundle = _fresh(tmp_path, monkeypatch, "src6")
    _populate(store, n=11)
    out = tmp_path / "b6.zip"
    bundle.export(out, include_space=False)
    data = bundle.inspect(out)
    assert data["titles"] == 11
    assert data["format"] == bundle.FORMAT_VERSION
    assert "verdicts" in data["note"]
