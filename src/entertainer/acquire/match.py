"""Deciding whether a file is the film that was asked for.

File names are the only evidence most sources give, and they lie in
predictable ways: remakes share titles, sequels share prefixes, samples and
trailers carry the full name. Year agreement is therefore required whenever
both sides have one, and a name that parses as an episode is refused — a
series file dropped into a film library is worse than nothing.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher

VIDEO_EXTS = frozenset({"mkv", "mp4", "avi", "m4v", "mov", "webm", "ts", "mpg", "mpeg", "ogv"})
_JUNK = re.compile(r"\b(sample|trailer|teaser|promo|preview)\b", re.I)

# 1080p first: 2160p roughly quadruples disk for a difference most home screens
# do not show, and the disk budget is the scarcer resource for an unsupervised agent.
_RESOLUTION_RANK = {"1080p": 3.0, "2160p": 2.0, "720p": 2.0, "576p": 1.0, "480p": 0.5}


@dataclass(frozen=True)
class Target:
    title: str
    year: int | None = None
    runtime_min: int | None = None
    alt_titles: tuple[str, ...] = ()
    imdb_id: str | None = None
    language: str | None = None  # English name, e.g. "Tamil"; dubs are the common wrong file

    @property
    def folder(self) -> str:
        name = re.sub(r'[\\/:*?"<>|]', "", self.title).strip()
        return f"{name} ({self.year})" if self.year else name


@dataclass
class Candidate:
    source: str
    label: str
    size: int | None
    ref: object = field(repr=False)
    ext: str = ""
    score: float = 0.0
    why: str = ""


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"&", " and ", s.lower())
    s = re.sub(r"^(the|a|an)\s+", "", s)
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def parse(label: str) -> dict:
    from guessit import guessit

    return dict(guessit(label))


def score(target: Target, cand: Candidate) -> Candidate:
    """Fill ``cand.score``; zero means refused, with the reason in ``why``."""
    if _JUNK.search(cand.label):
        return _refuse(cand, "sample or trailer")
    info = parse(cand.label)
    ext = (cand.ext or str(info.get("container") or "")).lower().lstrip(".")
    if ext and ext not in VIDEO_EXTS:
        return _refuse(cand, f"not a video ({ext})")
    if info.get("type") == "episode":
        return _refuse(cand, "looks like an episode")

    got = norm(str(info.get("title") or ""))
    wanted = [norm(t) for t in (target.title, *target.alt_titles) if t]
    sim = max((SequenceMatcher(None, got, w).ratio() for w in wanted), default=0.0)
    if sim < 0.85:
        return _refuse(cand, f"title {got!r} is not {wanted[0]!r}")

    year = info.get("year")
    if target.year and year and abs(int(year) - target.year) > 1:
        return _refuse(cand, f"year {year} is not {target.year}")

    s = sim * 10
    s += 2.0 if (target.year and year) else -1.0
    s += _RESOLUTION_RANK.get(str(info.get("screen_size") or ""), 1.0)
    if str(info.get("video_codec") or "").upper() in ("H.265", "HEVC"):
        s += 0.5
    if target.language:
        # Groups index every dub under the original title; a Hindi dub of a
        # Tamil film matches on title and year and is still the wrong file.
        s += 2.0 if target.language.lower() in cand.label.lower() else -1.0
    if "Cam" in str(info.get("source") or "") or "Telesync" in str(info.get("source") or ""):
        s -= 5.0
    cand.ext = ext or cand.ext
    cand.score = s
    cand.why = f"{info.get('screen_size') or '?'} {info.get('video_codec') or ''}".strip()
    return cand


def _refuse(cand: Candidate, why: str) -> Candidate:
    cand.score, cand.why = 0.0, why
    return cand
