from __future__ import annotations

import asyncio


def test_enrichment_keeps_only_a_bounded_number_of_tasks(monkeypatch):
    from entertainer.data import tmdb

    active = 0
    peak = 0

    async def fake_one(_client, imdb_id, _params, _limiter, _keywords):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0)
        active -= 1
        return tmdb.EnrichResult(imdb_id=imdb_id)

    monkeypatch.setattr(tmdb, "_auth", lambda: ({}, {}))
    monkeypatch.setattr(tmdb, "_enrich_one", fake_one)
    result = asyncio.run(tmdb.enrich_async([f"tt{i:07d}" for i in range(9)], concurrency=2))
    assert len(result) == 9
    assert peak <= 2
