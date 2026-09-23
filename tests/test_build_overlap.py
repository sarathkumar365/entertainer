import operator
import threading
import time

import pytest

from entertainer import pipeline
from entertainer.build_events import Reporter
from entertainer.commands.build import _BackgroundStage
from entertainer.config import IMDB_FILES, PATHS
from entertainer.data import download


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTERTAINER_DATA_DIR", str(tmp_path))
    PATHS.ensure()
    return tmp_path


def test_sources_download_concurrently(data_dir, monkeypatch):
    active = 0
    peak = 0
    lock = threading.Lock()

    def slow_fetch(url, dest, progress=None):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"")
        return dest

    monkeypatch.setattr(download, "fetch", slow_fetch)
    ml = data_dir / "raw" / "ml-32m"
    ml.mkdir(parents=True)
    (ml / "ratings.csv").write_text("")

    imdb, _ = download.fetch_sources()

    assert [p.name for p in imdb] == list(IMDB_FILES)
    assert peak > 1


def test_catalogue_fingerprint_follows_inputs_and_settings(data_dir):
    settings = pipeline.BuildSettings()
    imdb = data_dir / "raw" / "imdb"
    imdb.mkdir(parents=True)
    for name in IMDB_FILES:
        (imdb / name).write_bytes(b"x")
    first = pipeline.catalogue_inputs(settings)

    assert pipeline.catalogue_inputs(settings) == first
    (imdb / (IMDB_FILES[0] + ".validator")).write_text('"v2"')
    assert pipeline.catalogue_inputs(settings) != first
    changed = pipeline.catalogue_inputs(settings)
    assert pipeline.catalogue_inputs(pipeline.BuildSettings(floor_scale=0.5)) != changed


def test_cf_is_current_only_after_a_recorded_run_with_the_same_inputs(data_dir):
    settings = pipeline.BuildSettings(cf_holdout=0)
    ml = data_dir / "raw" / "ml-32m"
    ml.mkdir(parents=True)
    (ml / "ratings.csv").write_text("userId,movieId,rating,timestamp\n")
    assert not pipeline.cf_is_current(settings)

    (PATHS.embeddings / "cf_factors.npy").write_bytes(b"")
    pipeline.record_cf_inputs(pipeline.cf_inputs(settings))
    assert pipeline.cf_is_current(settings)
    assert not pipeline.cf_is_current(pipeline.BuildSettings(cf_holdout=0, cf_iterations=3))

    (ml / "ratings.csv").write_text("userId,movieId,rating,timestamp\n1,2,3.0,0\n")
    assert not pipeline.cf_is_current(settings)


def test_cf_is_current_needs_the_holdout_when_one_was_asked_for(data_dir):
    settings = pipeline.BuildSettings(cf_holdout=10)
    (PATHS.embeddings / "cf_factors.npy").write_bytes(b"")
    pipeline.record_cf_inputs(pipeline.cf_inputs(settings))
    assert not pipeline.cf_is_current(settings)


def _stage(reporter, stage_id):
    return next(s for s in reporter.state["stages"] if s["id"] == stage_id)


def test_background_stage_reports_completion(tmp_path):
    reporter = Reporter(root=tmp_path)
    job = _BackgroundStage(reporter, "cf", operator.add, 1, 2)
    job.wait()
    assert _stage(reporter, "cf")["status"] == "complete"


def test_background_stage_failure_surfaces_on_wait(tmp_path):
    reporter = Reporter(root=tmp_path)
    job = _BackgroundStage(reporter, "cf", operator.truediv, 1, 0)
    with pytest.raises(RuntimeError, match="background stage failed"):
        job.wait()
    assert _stage(reporter, "cf")["status"] == "failed"


def test_a_standalone_factorisation_invalidates_the_setup_stamp(data_dir, monkeypatch):
    from entertainer.commands.build import data_cf
    from entertainer.models import cf

    settings = pipeline.BuildSettings(cf_holdout=0)
    (PATHS.embeddings / "cf_factors.npy").write_bytes(b"")
    pipeline.record_cf_inputs(pipeline.cf_inputs(settings))

    def stop():
        raise RuntimeError("stop before the fit")

    monkeypatch.setattr(cf, "load_ratings", stop)
    with pytest.raises(RuntimeError):
        data_cf(factors=64, iterations=1, holdout=0, signal="watched")
    assert not pipeline.cf_is_current(settings)


@pytest.mark.parametrize("dry_run, kept", [(False, False), (True, True)])
def test_a_prune_outside_setup_forgets_the_catalogue_inputs(data_dir, dry_run, kept):
    from entertainer import store
    from entertainer.data import catalog

    with store.session() as con:
        store.set_meta(con, pipeline.CATALOGUE_INPUTS_KEY, "abc")
    catalog.prune_by_language(dry_run=dry_run)
    with store.session() as con:
        assert (store.get_meta(con, pipeline.CATALOGUE_INPUTS_KEY) == "abc") is kept
