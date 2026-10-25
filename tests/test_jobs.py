"""Jobs and the worker: records, pauses, wrong files, restarts, cancel, the request verdict.

Driven with a fake source and a one-minute ffmpeg clip, as ``test_acquire`` is,
so nothing here reaches Telegram or TMDB.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

pytest.importorskip("guessit")

from entertainer.acquire.match import Candidate, Target  # noqa: E402
from entertainer.agent import jobs as J  # noqa: E402
from entertainer.agent import run, worker  # noqa: E402
from entertainer.config import AcquireSettings  # noqa: E402
from entertainer.errors import AcquireBusy, CannotAcquire, EntertainerError  # noqa: E402

SHORT = Target("Vikram", 2022, 1, imdb_id="tt9179430")
GOOD = "Vikram.2022.1080p.x265.HEVC.mp4"
NEXT = "Vikram.2022.720p.x265.HEVC.mp4"


class Fake:
    name = "fake"

    def __init__(self, labels, clip: Path, delay: float = 0.0):
        self.labels, self.clip, self.delay, self.fetched = labels, clip, delay, []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass

    async def search(self, target):
        return [Candidate("fake", lbl, 1000, lbl, ext="mp4") for lbl in self.labels]

    async def fetch(self, cand, dest, progress=None):
        self.fetched.append(cand.label)
        dest.write_bytes(b"x")
        if progress:
            progress(1, 1000)
        await asyncio.sleep(self.delay)
        shutil.copy(self.clip, dest)
        return dest


@pytest.fixture
def clip(tmp_path) -> Path:
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    out = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=d=60:s=64x64",
                    "-pix_fmt", "yuv420p", str(out)], check=True)
    return out


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A library, a jobs directory, and the agent pointed at a fake source."""
    monkeypatch.setenv("ENTERTAINER_DATA_DIR", str(tmp_path / "data"))
    lib = tmp_path / "Movies"
    lib.mkdir()
    s = AcquireSettings(
        enabled=True, library_dir=lib, max_gb=3.0, reserve_gb=0.0, sources=("fake",),
        tg_api_id=None, tg_api_hash=None, tg_channels=(), tg_groups=(),
        tg_session=tmp_path / "tg" / "s.session", jellyfin_url=None, jellyfin_key=None,
    )
    box: dict = {}

    @asynccontextmanager
    async def fake_open(_settings, _names=None, on_join=None):
        src = box["source"]
        if on_join:
            on_join("https://t.me/+abc", "777")
        yield {src.name: src}

    monkeypatch.setattr(run, "open_sources", fake_open)
    monkeypatch.setattr(worker, "resolve", lambda job: SHORT)
    monkeypatch.setattr(worker, "catalogue_id", lambda imdb: None)
    jobs = J.Jobs(tmp_path / "jobs")

    def use(source):
        box["source"] = source
        return source

    return s, jobs, use


def _run(env, **kw) -> J.Job:
    s, jobs, _ = env
    job = jobs.create({"imdb_id": SHORT.imdb_id, "via": "test"})
    return worker.run_one(jobs, job.id, s, **kw)


def test_a_job_records_its_search_its_pick_and_where_it_filed(env, clip):
    env[2](Fake([GOOD], clip))
    job = _run(env)
    assert job.state == J.FILED and Path(job.filed["path"]).exists()
    assert job.folder == "Vikram (2022)"
    assert [c["label"] for c in job.candidates] == [GOOD]
    assert [c["label"] for c in job.pick] == [GOOD]
    assert "ref" not in job.candidates[0]


def test_every_join_is_recorded_against_its_job(env, clip):
    env[2](Fake([GOOD], clip))
    assert _run(env).joins == [{"link": "https://t.me/+abc", "chat": "777", "left": False}]


def test_an_unclear_pick_pauses_and_the_choice_resumes_it(env, clip):
    s, jobs, use = env
    use(Fake(["Vikram 720p", "Vikram 480p"], clip))  # no year: ask
    job = _run(env)
    assert job.state == J.PAUSED and len(job.offer) == 2 and job.asked, job.reason
    job = worker.Worker(jobs, lambda: s).choose(job.id, 1)
    assert job.state == J.QUEUED
    job = worker.run_one(jobs, job.id, s)
    assert job.state == J.FILED and job.filed["label"] == "Vikram 480p"


def test_a_choice_that_is_no_longer_offered_fails_with_a_next_step(env, clip):
    s, jobs, use = env
    use(Fake(["Vikram 720p", "Vikram 480p"], clip))
    job = _run(env)
    worker.Worker(jobs, lambda: s).choose(job.id, 0)
    use(Fake(["Vikram 480p"], clip))  # the bot's list moved on
    job = worker.run_one(jobs, job.id, s)
    assert job.state == J.FAILED and "Ask for the film again" in job.reason


def test_wrong_file_deletes_the_copy_and_never_tries_it_again(env, clip):
    s, jobs, use = env
    src = use(Fake([GOOD, NEXT], clip))
    job = _run(env)
    first = Path(job.filed["path"])
    job = worker.Worker(jobs, lambda: s).wrong(job.id)
    assert not first.exists() and job.state == J.QUEUED and job.wrong == [GOOD]
    job = worker.run_one(jobs, job.id, s)
    assert job.state == J.FILED and job.filed["label"] == NEXT
    assert src.fetched == [GOOD, NEXT]


def test_a_job_killed_mid_run_comes_back_failed_with_its_partial_file_gone(env):
    s, jobs, _ = env
    job = jobs.create({"imdb_id": SHORT.imdb_id})
    jobs.append(job.id, "target", folder=SHORT.folder, imdb_id=SHORT.imdb_id, item_id=None)
    dead = subprocess.Popen(["true"])
    dead.wait()
    jobs.state(job.id, J.DOWNLOADING, pid=dead.pid)
    s.staging.mkdir(parents=True)
    part = s.staging / f"{SHORT.folder}.part"
    part.write_bytes(b"half a film")
    worker.Worker(jobs, lambda: s).recover()
    assert jobs.get(job.id).state == J.FAILED and not part.exists()


def test_a_job_another_live_process_is_running_is_left_alone(env):
    s, jobs, _ = env
    job = jobs.create({"imdb_id": SHORT.imdb_id})
    jobs.state(job.id, J.DOWNLOADING, pid=os.getpid())
    worker.Worker(jobs, lambda: s).recover()
    assert jobs.get(job.id).state == J.DOWNLOADING


def test_a_job_over_its_time_budget_fails_and_leaves_no_partial_file(env, clip):
    s, _, use = env
    use(Fake([GOOD], clip, delay=5))
    job = _run(env, budget_s=0.3)
    assert job.state == J.FAILED and "gave up" in job.reason
    assert not any(s.staging.iterdir())


def test_cancelling_a_running_job_stops_the_download(env, clip):
    s, jobs, use = env
    use(Fake([GOOD], clip, delay=10))
    w = worker.Worker(jobs, lambda: s, idle_s=0.05)
    w.start()
    try:
        job = w.submit({"imdb_id": SHORT.imdb_id})
        _until(lambda: jobs.get(job.id).state == J.DOWNLOADING)
        w.cancel(job.id)
        _until(lambda: jobs.get(job.id).terminal)
    finally:
        w.stop()
    assert jobs.get(job.id).state == J.CANCELLED
    assert not any(s.staging.iterdir())


def test_the_worker_runs_queued_jobs_and_reports_progress(env, clip):
    s, jobs, use = env
    use(Fake([GOOD], clip, delay=0.5))
    w = worker.Worker(jobs, lambda: s, idle_s=0.05)
    w.start()
    try:
        job = w.submit({"imdb_id": SHORT.imdb_id})
        _until(lambda: job.id in w.progress)
        _until(lambda: jobs.get(job.id).terminal)
    finally:
        w.stop()
    assert jobs.get(job.id).state == J.FILED


def test_asking_twice_for_the_same_film_is_one_job(env):
    s, jobs, _ = env
    w = worker.Worker(jobs, lambda: s)
    assert w.submit({"imdb_id": "tt1"}).id == w.submit({"imdb_id": "tt1"}).id


def test_a_machine_that_cannot_acquire_refuses_before_queueing(env):
    s, jobs, _ = env
    off = AcquireSettings(**{**s.__dict__, "enabled": False})
    with pytest.raises(CannotAcquire, match="switched off"):
        worker.Worker(jobs, lambda: off).submit({"imdb_id": "tt1"})
    assert jobs.all() == []


def test_a_pause_unanswered_for_a_day_is_cancelled(env):
    s, jobs, _ = env
    job = jobs.create({"imdb_id": "tt1"})
    jobs.state(job.id, J.PAUSED, "unclear")
    worker.Worker(jobs, lambda: s, pause_s=-1)._expire_pauses()
    assert jobs.get(job.id).state == J.CANCELLED


def test_only_a_paused_job_takes_a_choice_and_only_a_filed_one_is_wrong(env):
    s, jobs, _ = env
    w = worker.Worker(jobs, lambda: s)
    job = jobs.create({"imdb_id": "tt1"})
    with pytest.raises(EntertainerError, match="not waiting"):
        w.choose(job.id, 0)
    with pytest.raises(EntertainerError, match="Only a filed copy"):
        w.wrong(job.id)


def test_a_job_id_cannot_name_a_file_outside_the_jobs_directory(env):
    _, jobs, _ = env
    jobs.root.mkdir(parents=True, exist_ok=True)
    (jobs.root.parent / "secret.jsonl").write_text("{}")
    with pytest.raises(EntertainerError, match="No download job"):
        jobs.get("../secret")


def test_a_torn_last_line_does_not_lose_the_job(env):
    _, jobs, _ = env
    job = jobs.create({"imdb_id": "tt1"})
    jobs.state(job.id, J.SEARCHING)
    with open(jobs.root / f"{job.id}.jsonl", "a") as f:
        f.write('{"at": "2026-')
    assert jobs.get(job.id).state == J.SEARCHING


def test_a_requested_title_leaves_the_recommendations(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTERTAINER_DATA_DIR", str(tmp_path))
    from entertainer import config, store

    config.PATHS.ensure()
    with store.session() as con:
        con.execute("INSERT INTO titles (item_id, imdb_id, kind, title) "
                    "VALUES (7, 'tt7', 'movie', 'Seven')")
    jobs = J.Jobs(tmp_path / "jobs")
    job = jobs.create({"item_id": 7})
    jobs.append(job.id, "target", folder="Seven", imdb_id="tt7", item_id=7)
    worker.log_request(jobs, jobs.get(job.id))
    worker.log_request(jobs, jobs.get(job.id))  # once per job, however often it runs
    with store.session(read_only=True) as con:
        assert 7 in store.interacted(con)
        n, src = con.execute("SELECT count(*), any_value(source) FROM events "
                             "WHERE kind = 'request'").fetchone()
    assert (n, src) == (1, "agent")


def _until(cond, timeout: float = 10.0) -> None:
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return
        time.sleep(0.02)
    raise AssertionError("timed out")


def test_leaving_joined_channels_undoes_only_the_recorded_joins(env, clip):
    s, jobs, use = env
    use(Fake([GOOD], clip))
    _run(env)
    w = worker.Worker(jobs, lambda: s)
    assert w.joined() == ["777"]
    asked = []

    async def fake_leave(_settings, chats):
        asked.append(chats)
        return chats, {}

    assert w.leave_joined(fake_leave) == "Left 1 channel."
    assert asked == [["777"]] and w.joined() == []
    assert w.leave_joined(fake_leave) == "No joined channels to leave."
    assert jobs.all()[0].joins[0]["left"]


def test_a_channel_that_will_not_be_left_stays_on_the_list(env, clip):
    s, jobs, use = env
    use(Fake([GOOD], clip))
    _run(env)
    w = worker.Worker(jobs, lambda: s)

    async def stubborn(_settings, chats):
        return [], {chats[0]: "ChannelPrivateError: gone"}

    assert "could not be left" in w.leave_joined(stubborn)
    assert w.joined() == ["777"]


def _recorded(jobs, labels, **outcome):
    """A job as the worker leaves it, without running one."""
    job = jobs.create({"imdb_id": SHORT.imdb_id})
    jobs.append(job.id, "target", folder=SHORT.folder, imdb_id=SHORT.imdb_id, item_id=None,
                film=worker.film(SHORT))
    jobs.append(job.id, "plan", candidates=[
        {"source": "t", "label": lbl, "size": 1000, "ext": ""} for lbl in labels], errors={})
    if "filed" in outcome:
        jobs.append(job.id, "filed", path="/x", label=outcome["filed"], source="t")
    if "wrong" in outcome:
        jobs.append(job.id, "wrong", label=outcome["wrong"], source="t")
    if "confirm" in outcome:
        jobs.append(job.id, "confirm", label=outcome["confirm"])
    return job


def test_replay_scores_recorded_jobs_against_the_users_labels(env, monkeypatch):
    from entertainer.agent import replay as R

    _, jobs, _ = env
    _recorded(jobs, [GOOD, NEXT], filed=GOOD)                       # right
    _recorded(jobs, [GOOD, NEXT], wrong=GOOD)                       # the rule repeats a mistake
    _recorded(jobs, ["Vikram 720p", "Vikram 480p"], confirm="Vikram 480p")  # a seed: pauses
    _recorded(jobs, [GOOD])                                         # no label either way
    _recorded(jobs, ["Vikram.1986.720p.mkv"])                       # nothing usable
    r = R.replay(jobs, max_gb=3)
    assert r.totals == {"right": 1, "wrong": 1, "paused": 1, "unlabelled": 1, "nothing": 1}


def test_replay_needs_no_network(env, monkeypatch):
    """The point of recording candidates: nothing is searched again."""
    from entertainer.agent import replay as R

    _, jobs, _ = env
    monkeypatch.setattr(run, "open_sources", None)
    monkeypatch.setattr(worker, "resolve", None)
    _recorded(jobs, [GOOD], filed=GOOD)
    assert R.replay(jobs, max_gb=3).totals["right"] == 1


def test_a_baseline_is_saved_and_read_back(env):
    from entertainer.agent import replay as R

    _, jobs, _ = env
    _recorded(jobs, [GOOD, NEXT], wrong=GOOD)
    R.save_baseline(R.replay(jobs, max_gb=3))
    assert R.load_baseline()["totals"]["wrong"] == 1


def test_jellyfin_check_says_what_is_wrong(env, monkeypatch):
    import httpx

    from entertainer.agent.library import check_jellyfin

    s, _, _ = env
    assert check_jellyfin(s) == (False, check_jellyfin(s)[1])
    assert "not configured" in check_jellyfin(s)[1]
    on = AcquireSettings(**{**s.__dict__, "jellyfin_url": "http://jf:8096", "jellyfin_key": "k"})

    def answer(url, headers, timeout):
        assert url == "http://jf:8096/System/Info" and headers == {"X-Emby-Token": "k"}
        return httpx.Response(200, json={"Version": "10.10.3", "ServerName": "home"},
                              request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", answer)
    assert check_jellyfin(on) == (True, "Jellyfin 10.10.3 on home")

    def down(url, headers, timeout):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", down)
    ok, msg = check_jellyfin(on)
    assert not ok and "did not answer" in msg


def test_a_job_the_terminal_runs_is_never_taken_by_the_apps_worker(env, clip):
    s, jobs, use = env
    use(Fake([GOOD], clip))
    job = jobs.create({"imdb_id": SHORT.imdb_id}, run_here=True)
    assert jobs.get(job.id).state == J.SEARCHING
    assert worker.Worker(jobs, lambda: s)._next() is None


def test_one_job_runs_in_one_process_only(env, clip):
    s, jobs, use = env
    use(Fake([GOOD], clip))
    job = jobs.create({"imdb_id": SHORT.imdb_id})
    with jobs.claim(job.id), pytest.raises(AcquireBusy, match="already running"):
        worker.run_one(jobs, job.id, s)
    assert worker.run_one(jobs, job.id, s).state == J.FILED


def test_a_job_running_in_another_process_cannot_be_cancelled_from_here(env):
    s, jobs, _ = env
    job = jobs.create({"imdb_id": SHORT.imdb_id})
    sleeper = subprocess.Popen(["sleep", "5"])
    try:
        jobs.state(job.id, J.DOWNLOADING, pid=sleeper.pid)
        with pytest.raises(EntertainerError, match="running in a terminal"):
            worker.Worker(jobs, lambda: s).cancel(job.id)
    finally:
        sleeper.kill()
