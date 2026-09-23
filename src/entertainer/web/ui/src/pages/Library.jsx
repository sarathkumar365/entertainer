import { useCallback } from "react";

import { api } from "../api";
import Unavailable from "../components/Unavailable";
import { useAsync } from "../useAsync";

/**
 * What you have saved, watched, and rated.
 *
 * A save is not a verdict. It says you intend to watch something, never that
 * you liked it, so nothing on the Saved shelf has taught the model anything
 * — it only keeps those titles out of future recommendations.
 */
const SHELVES = [
  ["saved", "Saved", "kept out of recommendations, but not a verdict"],
  ["watched", "Watched", "seen, with no opinion recorded"],
  ["rated", "Rated", "these are what the model learned from"],
];

function Shelf({ id, title, note, items, onAction }) {
  return (
    <section className="shelf">
      <header>
        <span className="label">{title}</span>
        <span className="label num">{items.length}</span>
      </header>
      <p className="label shelf-note">{note}</p>
      {items.length === 0 ? (
        <p className="shelf-empty label">nothing here yet</p>
      ) : (
        <ul className="shelf-list">
          {items.map((item) => (
            <li key={item.item_id} className="enter">
              {item.poster ? <img src={item.poster} alt="" loading="lazy" /> : null}
              <div>
                <div className="shelf-title">{item.title}</div>
                <div className="label">
                  {[item.year, item.language_name].filter(Boolean).join(" · ")}
                </div>
              </div>
              {id === "saved" ? (
                <div className="shelf-actions">
                  <button type="button" onClick={() => onAction(item.item_id, "watched")}>
                    Watched
                  </button>
                  <button type="button" onClick={() => onAction(item.item_id, "remove")}>
                    Remove
                  </button>
                </div>
              ) : null}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

export default function Library() {
  const { data, error, loading, reload } = useAsync(() => api.library(), []);

  const act = useCallback(
    async (itemId, action) => {
      await api.libraryAction(itemId, action);
      reload();
    },
    [reload],
  );

  if (error) {
    return (
      <div className="page">
        <Unavailable error={error} />
      </div>
    );
  }

  return (
    <div className="page">
      <header>
        <span className="label">Library</span>
        <h1>What you have kept</h1>
        <p>
          Saving is reversible and teaches the model nothing — it only stops a
          title being recommended again. A verdict is the opposite: it trains
          the model, and the only way to change one is to record another.
        </p>
      </header>

      {loading ? <p className="label">loading…</p> : null}

      {data
        ? SHELVES.map(([id, title, note]) => (
            <Shelf key={id} id={id} title={title} note={note} items={data[id]} onAction={act} />
          ))
        : null}
    </div>
  );
}
