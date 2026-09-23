import { useState } from "react";

import { VERDICTS } from "../api";
import Distribution from "./Distribution";

const FACTS = (item) =>
  [
    item.year,
    item.language_name,
    item.kind === "tv" ? "series" : null,
    item.runtime ? `${item.runtime}m` : null,
    item.rating ? `IMDb ${item.rating.toFixed(1)}` : null,
  ].filter(Boolean);

/**
 * One title, with everything the payload already carries.
 *
 * The old grid threw away genres, directors, runtime and the overview even
 * though every one of them was in the response.
 */
export default function TitleCard({
  item,
  onVerdict,
  verdicts = VERDICTS,
  showPrediction = false,
  busy = false,
  onSave = null,
}) {
  const [leaving, setLeaving] = useState(null);

  const give = (verdict) => {
    // One verdict per card. The call is deferred so the card can animate
    // out, and without this guard a second click inside that window queued a
    // second timer — two rate events for one title, which then distorts the
    // prequential replay that walks the log in order.
    if (leaving) return;
    setLeaving(verdict);
    // Long enough to read as a departure, short enough not to be a wait.
    setTimeout(() => onVerdict?.(item, verdict), 160);
  };

  return (
    <article className={`card${leaving ? " card-leaving" : ""}`} data-verdict={leaving ?? undefined}>
      <div className={`card-art${item.poster ? "" : " card-art-empty"}`}>
        {item.poster ? (
          <img src={item.poster} alt="" loading="lazy" />
        ) : (
          <div className="card-art-blank label">no poster</div>
        )}
        {item.explored ? <span className="badge label">exploring</span> : null}
      </div>

      <div className="card-body">
        <h3 className="card-title">{item.title}</h3>
        {item.original_title ? <p className="card-original">{item.original_title}</p> : null}
        <p className="card-facts label">{FACTS(item).join(" · ")}</p>

        {item.genres?.length ? <p className="card-genres">{item.genres.join(", ")}</p> : null}
        {item.overview ? <p className="card-overview">{item.overview}</p> : null}

        {showPrediction && item.score != null ? (
          <div className="card-prediction">
            <div className="card-prediction-head">
              <span className="reading-sm">{item.score.toFixed(1)}</span>
              <span className="label">
                ± {(item.std ?? 0).toFixed(1)}
              </span>
            </div>
            <Distribution score={item.score} std={item.std} />
          </div>
        ) : null}

        {item.reasons?.length ? (
          <p className="card-reasons">
            <span className="label">close to </span>
            {item.reasons.map(([name]) => (
              <span key={name} className="chip">{name}</span>
            ))}
          </p>
        ) : null}

        {onSave ? (
          <button
            type="button"
            className="save"
            disabled={busy || leaving !== null}
            onClick={() => { if (!leaving) { setLeaving("save"); setTimeout(onSave, 160); } }}
          >
            Save for later
          </button>
        ) : null}

        <div className="verdicts">
          {verdicts.map((v) => (
            <button
              key={v.key}
              type="button"
              className="verdict"
              disabled={busy || leaving !== null}
              onClick={() => give(v.key)}
            >
              {v.label}
            </button>
          ))}
        </div>
      </div>
    </article>
  );
}
