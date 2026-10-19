"""Internet Archive as a source: public-domain and openly licensed films.

Small catalogue, mostly pre-1970, but fully automatable and free of any
licensing question — which also makes it the source the spike is verified
against end to end. Item titles are matched, not file names: archive file
names are arbitrary, so the label is built from the item's own metadata.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx

from .match import VIDEO_EXTS, Candidate, Target

API = "https://archive.org"

# Preference among the formats the Archive derives. The original upload is
# often a huge MPEG2 or an odd container; the h.264 derivative is what plays.
_FORMAT_RANK = {"h.264 hd": 4, "h.264": 3, "mpeg4": 2, "matroska": 2, "512kb mpeg4": 1}


class ArchiveSource:
    name = "archive"

    async def __aenter__(self):
        self._http = httpx.AsyncClient(timeout=60, follow_redirects=True)
        return self

    async def __aexit__(self, *exc):
        await self._http.aclose()

    async def search(self, target: Target) -> list[Candidate]:
        q = f'title:("{target.title}") AND mediatype:movies'
        if target.year:
            q += f" AND year:[{target.year - 1} TO {target.year + 1}]"
        r = await self._http.get(
            f"{API}/advancedsearch.php",
            params={"q": q, "fl[]": ["identifier", "title", "year", "downloads"],
                    "rows": 10, "sort[]": "downloads desc", "output": "json"},
        )
        r.raise_for_status()
        out: list[Candidate] = []
        for doc in r.json()["response"]["docs"]:
            out.extend(await self._files(doc, target))
        return out

    async def _files(self, doc: dict, target: Target) -> list[Candidate]:
        ident = doc["identifier"]
        meta = (await self._http.get(f"{API}/metadata/{ident}")).json()
        year = str(doc.get("year") or target.year or "")[:4]
        best = None
        for f in meta.get("files", []):
            ext = f["name"].rsplit(".", 1)[-1].lower()
            rank = _FORMAT_RANK.get(str(f.get("format", "")).lower(), 0)
            if ext not in VIDEO_EXTS or not rank:
                continue
            if best is None or rank > best[0]:
                best = (rank, f)
        if best is None:
            return []
        f = best[1]
        ext = f["name"].rsplit(".", 1)[-1].lower()
        return [Candidate(
            source="archive",
            label=f"{doc.get('title')} ({year}).{ext}",
            size=int(f["size"]) if f.get("size") else None,
            ref=f"{API}/download/{ident}/{f['name']}",
            ext=ext,
        )]

    async def fetch(self, cand: Candidate, dest: Path, progress: Callable | None = None) -> Path:
        async with self._http.stream("GET", str(cand.ref)) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length") or cand.size or 0)
            done = 0
            with dest.open("wb") as fh:
                async for chunk in r.aiter_bytes(1 << 20):
                    fh.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(done, total)
        return dest
