from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("guessit")

from entertainer.acquire import core  # noqa: E402
from entertainer.acquire.match import Candidate, Target, score  # noqa: E402
from entertainer.acquire.telegram import _candidate  # noqa: E402
from entertainer.errors import NotAcquired  # noqa: E402

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

    def __init__(self, labels, clip: Path):
        self.labels, self.clip, self.fetched = labels, clip, []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass

    async def search(self, target):
        return [Candidate("fake", lbl, 1000, lbl, ext="mp4") for lbl in self.labels]

    async def fetch(self, cand, dest, progress=None):
        self.fetched.append(cand.label)
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


def test_a_file_whose_length_disagrees_is_deleted_not_filed(tmp_path, clip):
    src = FakeSource(["Vikram.2022.1080p.mp4"], clip)  # one-minute file, 174-minute film
    with pytest.raises(NotAcquired):
        asyncio.run(core.acquire(VIKRAM, [src], tmp_path / "Movies", log=lambda _: None))
    assert src.fetched and not any((tmp_path / "Movies").rglob("*.mp4"))
    assert not any((tmp_path / ".staging").iterdir())


def test_a_verified_file_is_filed_the_way_jellyfin_expects(tmp_path, clip):
    short = Target("Vikram", 2022, 1)
    src = FakeSource(["Vikram.2022.1080p.mp4"], clip)
    got = asyncio.run(core.acquire(short, [src], tmp_path / "Movies", log=lambda _: None))
    assert got.path == tmp_path / "Movies" / "Vikram (2022)" / "Vikram (2022).mp4"


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
