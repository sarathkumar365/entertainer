import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { VERDICTS, api } from "../api";
import TitleCard from "../components/TitleCard";
import Unavailable from "../components/Unavailable";
import { useAsync } from "../useAsync";

/**
 * Two modes over one data layer.
 *
 * Grid is for scanning and comparing. Focus is for getting through a lot of
 * titles quickly, which is the thing that actually makes the model good —
 * the engine learns from verdicts and nothing else.
 */
const KEYS = { l: "love", i: "like", o: "ok", m: "meh", d: "dislike", h: "hate", n: "unseen" };

/**
 * Live-mode titles are not in the catalogue and have no item_id, so both the
 * "already answered" and "in flight" maps key on whichever identifier the
 * item actually has. Declared at module scope: referencing it from the
 * filter above its own `const` would be a temporal-dead-zone error.
 */
const keyOf = (item) => item.item_id ?? `tmdb:${item.tmdb_id}`;

export default function Rate() {
  const [mode, setMode] = useState("grid");
  const [page, setPage] = useState(0);
  const [years, setYears] = useState(4);
  const [kind, setKind] = useState("");
  const [minQuality, setMinQuality] = useState(0);
  const [weights, setWeights] = useState(null);
  const [done, setDone] = useState({});
  // Keyed per title rather than one page-wide flag: a slow request used to
  // disable the verdict buttons on every card, which is the opposite of what
  // focus mode is for.
  const [busy, setBusy] = useState({});
  const [toast, setToast] = useState(null);
  //: catalogue item_id -> the `tmdb:` key its card was filed under.
  const placed = useRef(new Map());

  const { data: languages } = useAsync(() => api.languages(), []);

  useEffect(() => {
    if (languages && weights === null) {
      setWeights(Object.fromEntries(languages.languages.map((l) => [l.code, l.weight])));
    }
  }, [languages, weights]);

  const langParam = useMemo(
    () =>
      Object.entries(weights ?? {})
        .filter(([, w]) => w > 0)
        .map(([code, w]) => `${code}:${w}`)
        .join(","),
    [weights],
  );

  const { data, error, loading, reload } = useAsync(
    () => api.feed({ years, limit: 60, page, kind, min_quality: minQuality, langs: langParam }),
    [years, page, kind, minQuality, langParam],
  );

  const items = (data?.items ?? []).filter((i) => !(keyOf(i) in done));

  const give = useCallback(async (item, verdict) => {
    const key = keyOf(item);
    setDone((prev) => ({ ...prev, [key]: verdict }));
    setBusy((prev) => ({ ...prev, [key]: true }));
    try {
      if (item.external) {
        // Live mode: the title is not in the catalogue yet, so it has to be
        // pulled in before there is anything to attach a verdict to. The
        // response carries the catalogue row it became — remembered, because
        // undo reports the catalogue id and would otherwise have no way back
        // to the `tmdb:` key this card was filed under.
        const added = await api.add({ tmdb_id: item.tmdb_id, kind: item.kind, verdict });
        if (added?.item?.item_id != null) {
          placed.current.set(added.item.item_id, key);
        }
      } else {
        await api.rate({ item_id: item.item_id, verdict });
      }
      setToast(verdict === "unseen" ? "noted" : verdict);
    } catch (err) {
      setToast(err.message);
      setDone((prev) => {
        const next = { ...prev };
        delete next[key];
        return next;
      });
    } finally {
      setBusy((prev) => {
        const next = { ...prev };
        delete next[key];
        return next;
      });
    }
  }, []);

  const undo = useCallback(async () => {
    const result = await api.undo();
    if (result.ok) {
      setToast(`undid ${result.title ?? "that"}`);
      setDone((prev) => {
        const next = { ...prev };
        delete next[result.item_id];
        // A live-mode title was filed under `tmdb:<id>` before it had a
        // catalogue id. Undo reports the catalogue id, so without this the
        // card stays hidden after an undo that actually succeeded.
        const live = placed.current.get(result.item_id);
        if (live) {
          delete next[live];
          placed.current.delete(result.item_id);
        }
        return next;
      });
    }
  }, []);

  // Focus mode is keyboard-driven; that is the point of it.
  useEffect(() => {
    if (mode !== "focus") return undefined;
    const onKey = (event) => {
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      if (event.key === "u") {
        undo();
        return;
      }
      const verdict = KEYS[event.key.toLowerCase()];
      if (verdict && items[0]) give(items[0], verdict);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [mode, items, give, undo]);

  useEffect(() => {
    if (!toast) return undefined;
    const timer = setTimeout(() => setToast(null), 1600);
    return () => clearTimeout(timer);
  }, [toast]);

  const toggleLanguage = (code) =>
    setWeights((prev) => ({ ...prev, [code]: prev[code] > 0 ? 0 : 1 }));

  const setLanguageWeight = (code, value) =>
    setWeights((prev) => ({ ...prev, [code]: value }));

  return (
    <div className="page rate">
      <header className="rate-head">
        <div>
          <span className="label">Rate</span>
          <h1>Tell it what you have seen</h1>
        </div>
        <div className="mode-switch">
          <button type="button" className={mode === "grid" ? "on" : ""} onClick={() => setMode("grid")}>
            Grid
          </button>
          <button type="button" className={mode === "focus" ? "on" : ""} onClick={() => setMode("focus")}>
            Focus
          </button>
        </div>
      </header>

      <details className="filters">
        <summary className="label">Filters and languages</summary>
        <div className="filter-row">
          <label className="label">
            Since
            <input type="number" min="1" max="60" value={years}
                   onChange={(e) => { setPage(0); setYears(Number(e.target.value)); }} />
            <span>years ago</span>
          </label>
          <label className="label">
            Kind
            <select value={kind} onChange={(e) => { setPage(0); setKind(e.target.value); }}>
              <option value="">anything</option>
              <option value="movie">films</option>
              <option value="tv">series</option>
            </select>
          </label>
          <label className="label">
            Minimum quality
            <input type="range" min="0" max="1" step="0.05" value={minQuality}
                   onChange={(e) => { setPage(0); setMinQuality(Number(e.target.value)); }} />
            <span className="num">{minQuality.toFixed(2)}</span>
          </label>
        </div>

        <div className="languages">
          {(languages?.languages ?? []).map((l) => (
            <div key={l.code} className={`lang${(weights?.[l.code] ?? 0) > 0 ? " lang-on" : ""}`}>
              <button type="button" onClick={() => toggleLanguage(l.code)}>
                {l.name}
                {l.titles ? <span className="num"> {l.titles.toLocaleString()}</span> : null}
              </button>
              <input
                type="range" min="0" max="1" step="0.05"
                value={weights?.[l.code] ?? 0}
                onChange={(e) => setLanguageWeight(l.code, Number(e.target.value))}
                aria-label={`${l.name} weight`}
              />
            </div>
          ))}
        </div>
        <p className="label">
          Weights are continuous — half a language still appears, just less often.
        </p>
      </details>

      {error ? <Unavailable error={error} /> : null}
      {loading ? <p className="label">loading…</p> : null}

      {mode === "grid" ? (
        <div className="grid grid-tight">
          {items.map((item, i) => (
            <div key={item.item_id ?? item.tmdb_id} className="enter" style={{ animationDelay: `${i * 18}ms` }}>
              <TitleCard item={item} onVerdict={give} busy={!!busy[keyOf(item)]} />
            </div>
          ))}
        </div>
      ) : (
        <div className="focus">
          {items[0] ? (
            <>
              <div className="focus-card enter" key={items[0].item_id}>
                <TitleCard item={items[0]} onVerdict={give} busy={!!busy[keyOf(items[0])]} />
              </div>
              <ul className="focus-keys">
                {Object.entries(KEYS).map(([key, verdict]) => (
                  <li key={key}>
                    <kbd>{key}</kbd>
                    <span className="label">{VERDICTS.find((v) => v.key === verdict)?.label}</span>
                  </li>
                ))}
                <li><kbd>u</kbd><span className="label">Undo</span></li>
              </ul>
            </>
          ) : (
            <div className="empty">Nothing left in this slice.</div>
          )}
        </div>
      )}

      {!loading && items.length === 0 && !error ? (
        <div className="empty">
          <p>Nothing left here.</p>
          <button type="button" className="cta" onClick={() => setPage((p) => p + 1)}>More</button>
        </div>
      ) : null}

      {items.length > 0 && mode === "grid" ? (
        <div className="more">
          <button type="button" className="cta cta-quiet" onClick={() => setPage((p) => p + 1)}>
            Load more
          </button>
          <button type="button" className="cta cta-quiet" onClick={reload}>Refresh</button>
        </div>
      ) : null}

      {toast ? (
        <div className="toast">
          <span>{toast}</span>
          <button type="button" onClick={undo}>undo</button>
        </div>
      ) : null}
    </div>
  );
}
