from __future__ import annotations

import asyncio
import multiprocessing
import shutil
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("guessit")

from entertainer.acquire.match import Candidate, Target, score  # noqa: E402
from entertainer.acquire.telegram import _candidate  # noqa: E402
from entertainer.agent import run  # noqa: E402
from entertainer.config import AcquireSettings, acquire_settings  # noqa: E402
from entertainer.errors import AcquireBusy, CannotAcquire, NotAcquired  # noqa: E402

VIKRAM = Target("Vikram", 2022, 174)


def _c(label: str) -> Candidate:
    return score(VIKRAM, Candidate("telegram", label, 2 << 30, None))


def test_telegram_style_names_match_and_rank_1080p_first():
    good = _c("Vikram.2022.1080p.WEB-DL.x265.HEVC.Tamil.mkv")
    ok = _c("Vikram (2022) 720p HDRip.mp4")
    assert good.score > ok.score > 0


@pytest.mark.parametrize("label, why", [
    ("Vikram.1986.720p.mkv", "year"),
    ("Vikram.Vedha.2017.1080p.mkv", "title"),
    ("Vikram.2022.Sample.mkv", "sample"),
    ("Vikram.S01E02.1080p.mkv", "episode"),
    ("Vikram.2022.1080p.srt", "not a video"),
    ("Vikram (2022) | Behind the Scenes + Deleted Scenes.mp4", "sample"),
])
def test_wrong_files_are_refused(label, why):
    c = _c(label)
    assert c.score == 0 and why in c.why


def test_caption_labels_files_named_after_the_channel():
    msg = SimpleNamespace(
        file=SimpleNamespace(name="@somechannel_4412.mkv", ext=".mkv",
                             mime_type="video/x-matroska", size=1),
        message="Vikram (2022) 1080p Tamil\nJoin us",
    )
    assert _candidate(msg).label == "Vikram (2022) 1080p Tamil"


def test_non_video_posts_are_skipped():
    msg = SimpleNamespace(
        file=SimpleNamespace(name="poster.jpg", ext=".jpg", mime_type="image/jpeg", size=1),
        message="",
    )
    assert _candidate(msg) is None


class FakeSource:
    name = "fake"

    def __init__(self, labels, clip: Path, sizes=None, delivered_ext=None, fail=None):
        self.labels, self.clip, self.fetched = labels, clip, []
        self.sizes = sizes or [1000] * len(labels)
        self.delivered_ext, self.fail = delivered_ext, fail

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass

    async def search(self, target):
        if self.fail:
            raise self.fail
        ext = "" if self.delivered_ext else "mp4"
        return [Candidate("fake", lbl, size, lbl, ext=ext)
                for lbl, size in zip(self.labels, self.sizes, strict=True)]

    async def fetch(self, cand, dest, progress=None):
        self.fetched.append(cand.label)
        if self.delivered_ext:
            cand.ext = self.delivered_ext
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


def _settings(tmp_path, **kw) -> AcquireSettings:
    lib = tmp_path / "mnt" / "Movies"
    lib.mkdir(parents=True, exist_ok=True)
    base = dict(
        enabled=True, library_dir=lib, max_gb=3.0, reserve_gb=0.0, sources=("fake",),
        tg_api_id=None, tg_api_hash=None, tg_channels=(), tg_groups=(),
        tg_session=tmp_path / "tg" / "acquire.session", jellyfin_url=None, jellyfin_key=None,
    )
    return AcquireSettings(**{**base, **kw})


def _get(settings, target, *srcs):
    @asynccontextmanager
    async def fake_open(_settings, _names=None):
        yield {s.name: s for s in srcs}

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(run, "open_sources", fake_open)
        return asyncio.run(run.get(target, settings, log=lambda _: None))


SHORT = Target("Vikram", 2022, 1)


def test_a_file_whose_length_disagrees_is_deleted_not_filed(tmp_path, clip):
    s = _settings(tmp_path)
    src = FakeSource(["Vikram.2022.1080p.mp4"], clip)  # one-minute file, 174-minute film
    with pytest.raises(NotAcquired):
        _get(s, VIKRAM, src)
    assert src.fetched and not any(s.library_dir.rglob("*.mp4"))
    assert not any(s.staging.iterdir())


def test_a_verified_file_is_filed_the_way_jellyfin_expects(tmp_path, clip):
    s = _settings(tmp_path)
    got = _get(s, SHORT, FakeSource(["Vikram.2022.1080p.mp4"], clip))
    assert got.path == s.library_dir / "Vikram (2022)" / "Vikram (2022).mp4"


def test_staging_stays_inside_the_library(tmp_path, clip):
    """The library is a mount on the server; its parent may be another disk."""
    s = _settings(tmp_path)
    _get(s, SHORT, FakeSource(["Vikram.2022.1080p.mp4"], clip))
    assert s.staging.parent == s.library_dir
    assert sorted(p.name for p in s.library_dir.parent.iterdir()) == ["Movies"]


def test_a_copy_too_big_for_the_disk_is_skipped_not_fatal(tmp_path, clip):
    s = _settings(tmp_path)
    src = FakeSource(["Vikram.2022.2160p.mp4", "Vikram.2022.720p.mp4"], clip,
                     sizes=[1 << 60, 1000])
    got = _get(s, SHORT, src)
    assert src.fetched == ["Vikram.2022.720p.mp4"] and got.path.exists()


def test_one_source_failing_to_search_does_not_sink_the_others(tmp_path, clip):
    s = _settings(tmp_path)
    broken = FakeSource([], clip, fail=RuntimeError("archive.org 503"))
    broken.name = "broken"
    got = _get(s, SHORT, broken, FakeSource(["Vikram.2022.1080p.mp4"], clip))
    assert got.path.exists()


def test_the_delivered_file_decides_the_extension(tmp_path, clip):
    """Bot button labels often carry no extension; the file the bot sends does."""
    s = _settings(tmp_path)
    got = _get(s, SHORT, FakeSource(["Vikram 2022 TAMIL 1080p"], clip, delivered_ext="mp4"))
    assert got.path.suffix == ".mp4"


def test_a_delivered_non_video_is_rejected(tmp_path, clip):
    s = _settings(tmp_path)
    with pytest.raises(NotAcquired):
        _get(s, SHORT, FakeSource(["Vikram 2022 TAMIL 1080p"], clip, delivered_ext="exe"))
    assert not any(s.library_dir.rglob("*.exe"))


def test_every_copy_over_the_cap_is_refused_with_the_smallest_named(tmp_path, clip):
    s = _settings(tmp_path, max_gb=1.0)
    with pytest.raises(NotAcquired, match="smallest is 2.00 GB"):
        _get(s, SHORT, FakeSource(["Vikram.2022.1080p.mp4"], clip, sizes=[2 << 30]))


def test_a_film_already_in_the_library_is_not_fetched_again(tmp_path, clip):
    s = _settings(tmp_path)
    (s.library_dir / "Vikram (2022)").mkdir()
    (s.library_dir / "Vikram (2022)" / "Vikram (2022).mkv").touch()
    src = FakeSource(["Vikram.2022.1080p.mp4"], clip)
    with pytest.raises(NotAcquired, match="already in the library"):
        _get(s, SHORT, src)
    assert not src.fetched


def test_a_machine_without_a_library_says_so(tmp_path):
    s = _settings(tmp_path, library_dir=None, enabled=False)
    with pytest.raises(CannotAcquire, match="switched off.*ENTERTAINER_LIBRARY_DIR"):
        run.ready(s, ["archive"])


def test_telegram_sources_need_a_logged_in_session(tmp_path):
    s = _settings(tmp_path, tg_api_id="1", tg_api_hash="x")
    with pytest.raises(CannotAcquire, match="ent acquire login"):
        run.ready(s, ["telegram-bot"])


def test_only_one_process_holds_the_telegram_session(tmp_path):
    from entertainer.acquire.telegram import session_lock

    s = _settings(tmp_path)
    s.tg_lock.parent.mkdir(parents=True)
    ctx = multiprocessing.get_context("spawn")
    held, release = ctx.Event(), ctx.Event()
    p = ctx.Process(target=_hold_lock, args=(s.tg_lock, held, release))
    p.start()
    try:
        assert held.wait(10)
        with pytest.raises(AcquireBusy), session_lock(s):
            pass
    finally:
        release.set()
        p.join(10)
    with session_lock(s):  # free again once the holder is gone
        pass


def _hold_lock(path, held, release):
    import fcntl
    import os

    fd = os.open(path, os.O_RDWR | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX)
    held.set()
    release.wait(10)


def test_settings_are_read_from_the_environment_on_every_call(monkeypatch, tmp_path):
    monkeypatch.setenv("ENTERTAINER_LIBRARY_DIR", str(tmp_path / "lib"))
    monkeypatch.setenv("ENTERTAINER_TG_CHANNELS", "@films")
    monkeypatch.delenv("ENTERTAINER_TG_GROUPS", raising=False)
    monkeypatch.setenv("ENTERTAINER_ACQUIRE", "OFF")
    s = acquire_settings()
    assert s.staging == tmp_path / "lib" / ".staging"
    assert s.tg_groups == ("@films",) and not s.enabled and s.max_gb == 3.0


def _btn(text, **typ):
    return SimpleNamespace(text=text, to_dict=lambda: {"text": text, "type": typ})


def _msg(*buttons, mid=7):
    return SimpleNamespace(id=mid, reply_markup=SimpleNamespace(
        rows=[SimpleNamespace(buttons=[b]) for b in buttons]))


def test_filter_bot_reply_yields_only_file_buttons():
    from entertainer.acquire.telegram import _file_buttons, _next_page

    reply = _msg(
        _btn("🎖️Latest Movie Updates", url="https://t.me/+abc"),
        _btn(" 902.91 MB ◾ Vikram 2022 TAMIL 720p x265 HEVC HSB", data=b"file#AAA#1"),
        _btn(" 2.18 GB ◾ Vikram 2022 TAMIL 1080p x265 HEVC HSB", data=b"file#BBB#1"),
        _btn("🔞Great videos", url="https://ads.example"),
        _btn("​", data=b"nxt_pg 1 2 Vikram 2022"),
    )
    got = _file_buttons("group", reply)
    assert [c.label for c in got] == ["Vikram 2022 TAMIL 720p x265 HEVC HSB",
                                      "Vikram 2022 TAMIL 1080p x265 HEVC HSB"]
    assert got[0].size == int(902.91 * 2**20) and got[1].ref == ("group", 7, b"file#BBB#1", None)
    assert _next_page(reply) == b"nxt_pg 1 2 Vikram 2022"


def test_only_telegram_join_links_are_followed():
    from entertainer.acquire.telegram import _join_links

    gate = _msg(
        _btn("Request to Join Channel", url="https://t.me/+DLMs364"),
        _btn("Try again", url="https://t.me/SomeBot?start=abc"),
        _btn("Earn money", url="https://shortener.example/x"),
    )
    gate.message, gate.file = "Please join below channel to get file!", None
    assert _join_links(gate) == ["https://t.me/+DLMs364"]
    promo = _msg(_btn("Moviebot Updates", url="https://t.me/+d-3RL7"))
    promo.message, promo.file = "Vikram 2022.mkv", SimpleNamespace(name="Vikram 2022.mkv")
    assert _join_links(promo) == []


def test_the_original_language_outranks_a_dub():
    tamil = Target("Vikram", 2022, 174, language="Tamil")
    orig = score(tamil, Candidate("t", "Vikram 2022 TAMIL 720p x265 HEVC", 1, None))
    dub = score(tamil, Candidate("t", "Vikram 2022 720p x265 Hindi", 1, None))
    assert orig.score > dub.score > 0
