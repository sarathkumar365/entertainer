import { useSearchParams } from "react-router-dom";

import { api } from "../api";
import TasteField from "../components/TasteField";
import Unavailable from "../components/Unavailable";
import { useAsync } from "../useAsync";

const VIEWS = [
  ["figure", "Where you sit"],
  ["readings", "What it learned"],
];

/**
 * Plain-language readings of the five side features, keyed on
 * `features.SIDE_FEATURE_NAMES`. A name missing here still renders, under its
 * raw name, rather than vanishing.
 */
const HABITS = {
  "consensus quality": {
    name: "audience rating",
    hint: "the IMDb rating, trusted more when more people have rated it",
    towards: ["titles audiences rate highly", "titles with lower audience ratings"],
  },
  "how widely seen": {
    name: "popularity",
    hint: "how many people have rated it at all",
    towards: ["widely seen, popular titles", "lesser-known titles"],
  },
  "release recency": {
    name: "release date",
    hint: "the year it came out",
    towards: ["newer releases", "older titles"],
  },
  runtime: {
    name: "length",
    hint: "running time, capped at five hours",
    towards: ["longer titles", "shorter titles"],
  },
  "is a series": {
    name: "films or series",
    hint: "whether it is a series rather than a film",
    towards: ["series", "films"],
  },
};

function habitSentence(feature, widest) {
  const habit = HABITS[feature.name];
  const share = Math.abs(feature.weight) / widest;
  if (!habit) return `${feature.name}: ${feature.weight >= 0 ? "leans for" : "leans against"}`;
  if (Math.abs(feature.weight) < 0.005 || share < 0.08) {
    return `No lean either way on ${habit.name} yet.`;
  }
  const strength = share < 0.33 ? "A slight" : share < 0.66 ? "A clear" : "A strong";
  return `${strength} lean towards ${habit.towards[feature.weight >= 0 ? 0 : 1]}.`;
}

function listTerms(terms, fallback) {
  const top = terms.slice(0, 3);
  return top.length ? top.join(", ") : fallback;
}

function Direction({ axis, strongest }) {
  const width = (Math.abs(axis.strength) / strongest) * 100;
  return (
    <section className="axis enter">
      <h3 className="axis-title">
        Pulls you towards <em>{listTerms(axis.towards.terms, "something it cannot name yet")}</em>
      </h3>
      <p className="axis-away">
        and away from {listTerms(axis.away.terms, "something it cannot name yet")}
      </p>
      <div className="axis-pull">
        <span className="label">how much this shapes your picks</span>
        <div className="meter"><i style={{ width: `${width}%` }} /></div>
      </div>
      <div className="poles">
        <div>
          <div className="label pole-towards">Typical of what you are pulled towards</div>
          <p className="pole-examples">{axis.towards.examples.join(", ") || "—"}</p>
        </div>
        <div>
          <div className="label">Typical of what you are pulled away from</div>
          <p className="pole-examples">{axis.away.examples.join(", ") || "—"}</p>
        </div>
      </div>
    </section>
  );
}

function Readings({ data }) {
  const side = data.side_features ?? [];
  const widest = Math.max(0.001, ...side.map((f) => Math.abs(f.weight)));
  const strongest = Math.max(0.001, ...data.axes.map((a) => Math.abs(a.strength)));

  return (
    <div className="taste-readings">
      <p className="taste-lede">
        Learned from <span className="num">{data.n_verdicts.toLocaleString()}</span> verdicts.
        Nobody told it any of this.
      </p>

      <h2>The directions your taste runs in</h2>
      <p className="taste-note">
        It describes every title with a few hundred numbers, each one a way titles differ.
        These are the {data.axes.length} that best explain what you liked and disliked,
        strongest first. Each is described by the words and titles most typical of its two
        ends. The figure does not draw them, because a few hundred directions do not fit on a
        flat picture.
      </p>

      {data.axes.length === 0 ? (
        <div className="empty">Not enough signal to describe the directions yet.</div>
      ) : (
        data.axes.map((axis) => <Direction key={axis.index} axis={axis} strongest={strongest} />)
      )}

      <hr className="hairline" />
      <h2>Simple habits it noticed</h2>
      <p className="taste-note">
        Five plain properties of every title. It started with no opinion on any of them.
      </p>
      <div className="surface">
        {side.map((feature) => (
          <div key={feature.name} className="surface-row">
            <div>
              <p className="surface-sentence">{habitSentence(feature, widest)}</p>
              <p className="surface-hint">
                {HABITS[feature.name]?.name ?? feature.name} — {HABITS[feature.name]?.hint ?? ""}
              </p>
            </div>
            <div className="surface-bar">
              <i
                className={feature.weight >= 0 ? "pos" : "neg"}
                style={{ width: `${(Math.abs(feature.weight) / widest) * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>

      <details className="taste-details">
        <summary className="label">model details</summary>
        <p>
          {data.capacity.rff
            ? `It uses straight-line trade-offs between properties, plus ${data.capacity.rff} extra random features that let it pick up tastes which are not a straight line.`
            : "It uses straight-line trade-offs between properties. It has not yet found enough evidence to justify anything more complex."}
        </p>
        <p>
          Evidence score (log Z) <span className="num">{data.capacity.log_evidence.toFixed(1)}</span>:
          how well this model size explains your verdicts. It is used to choose the size, and
          only compares sizes fitted to the same verdicts.
        </p>
        <p>
          Raw weights:{" "}
          {side.map((f) => (
            <span key={f.name} className="num taste-raw">
              {f.name} {f.weight >= 0 ? "+" : ""}{f.weight.toFixed(3)}
            </span>
          ))}
        </p>
      </details>
    </div>
  );
}

export default function Taste() {
  const [params, setParams] = useSearchParams();
  const view = params.get("view") === "readings" ? "readings" : "figure";
  const { data, error, loading } = useAsync(() => api.taste(6), []);
  const { data: mode } = useAsync(() => api.mode(), []);

  // Fewer than three verdicts: nothing to describe yet, but the figure can
  // still show you at the start. Checked by code, because other refusals are
  // 409 too and must show their own message.
  const early = error?.code === "not_enough_evidence";

  return (
    <div className="page">
      <header className="taste-head">
        <div>
          <span className="label">Taste</span>
          <h1>What it has worked out</h1>
        </div>
        <div className="mode-switch taste-tabs" role="tablist">
          {VIEWS.map(([value, label]) => (
            <button
              key={value}
              type="button"
              role="tab"
              aria-selected={view === value}
              className={view === value ? "on" : ""}
              onClick={() => setParams(value === "figure" ? {} : { view: value }, { replace: true })}
            >
              {label}
            </button>
          ))}
        </div>
      </header>

      {view === "figure" ? (
        <>
          <TasteField
            verdicts={data?.n_verdicts ?? null}
            pending={!data && !early}
            catalogue={mode?.catalogue ?? null}
          />
          {error && !early ? <Unavailable error={error} /> : null}
        </>
      ) : error ? (
        early ? (
          <div className="empty">Rate a few titles first. After three verdicts there is something to read here.</div>
        ) : (
          <Unavailable error={error} />
        )
      ) : loading || !data ? (
        <p className="label">reading the model…</p>
      ) : (
        <Readings data={data} />
      )}
    </div>
  );
}
