import { useEffect, useRef } from "react";

import { api } from "../api";
import Organism from "../components/Organism";
import { useAsync } from "../useAsync";

/**
 * Is it actually learning?
 *
 * Large monospace readings in a broken grid. No organism at rest and no
 * field — measurement is this screen's whole job. The organism appears only
 * while the audit is computing, because that genuinely takes a couple of
 * seconds and grows with the verdict count.
 */
/** Trailing mean, so the trend is visible through per-step noise. */
function smooth(values, window) {
  return values.map((_, i) => {
    const from = Math.max(0, i - window + 1);
    const slice = values.slice(from, i + 1);
    return slice.reduce((a, b) => a + b, 0) / slice.length;
  });
}

function Curve({ curve }) {
  if (!curve?.steps?.length) return null;

  const width = 640;
  const height = 210;
  const pad = { l: 46, r: 16, t: 14, b: 30 };
  const n = curve.steps.length;

  // A single verdict's error swings the whole scale, so the raw series is
  // unreadable as a line on its own. It stays, faintly, because smoothing
  // away the evidence and showing only the trend would be the dishonest
  // version — the window is stated in the caption.
  const window = Math.max(5, Math.round(n / 12));
  const model = smooth(curve.absolute_error, window);
  const baseline = smooth(curve.baseline_error, window);
  const maxError = Math.max(0.001, ...curve.absolute_error, ...curve.baseline_error);

  const x = (i) => pad.l + (i / Math.max(n - 1, 1)) * (width - pad.l - pad.r);
  const y = (v) => pad.t + (1 - v / maxError) * (height - pad.t - pad.b);
  const line = (values) =>
    values.map((v, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");

  const last = n - 1;
  const modelPath = useRef(null);

  useEffect(() => {
    const path = modelPath.current;
    if (!path) return;
    const length = path.getTotalLength();
    path.style.strokeDasharray = `${length}`;
    path.style.setProperty("--draw", `${length}px`);
  });

  return (
    <figure className="curve">
      <svg viewBox={`0 0 ${width} ${height}`} role="img"
           aria-label="Prediction error over the verdict history, against a running-average baseline">
        <line x1={pad.l} y1={height - pad.b} x2={width - pad.r} y2={height - pad.b} className="axis-line" />
        <line x1={pad.l} y1={pad.t} x2={pad.l} y2={height - pad.b} className="axis-line" />
        <text x={pad.l - 8} y={pad.t + 8} className="tick" textAnchor="end">
          {(maxError * 10).toFixed(1)}
        </text>
        <text x={pad.l - 8} y={height - pad.b} className="tick" textAnchor="end">0</text>
        <text x={pad.l} y={height - 10} className="tick">{curve.steps[0]}</text>
        <text x={width - pad.r} y={height - 10} className="tick" textAnchor="end">
          {curve.steps[last]}
        </text>

        <path d={line(curve.absolute_error)} className="curve-raw" />
        <path d={line(baseline)} className="curve-baseline" />
        {/* Drawn in with CSS rather than SMIL: an <animate> element keeps
            the document from ever reporting itself as settled, which breaks
            anything waiting for a stable frame.

            The dash length is measured from the path rather than guessed. A
            fixed value hides everything past it, and pathLength="1" is not
            honoured for CSS stroke-dasharray here — it computes to 1px, so
            the line came out dotted. */}
        <path ref={modelPath} d={line(model)} className="curve-model" />
        <circle cx={x(last)} cy={y(model[last])} r="3" className="curve-end" />
      </svg>
      <figcaption className="label">
        <span className="key key-model">model error</span>
        <span className="key key-base">running-average baseline</span>
        <span className="key key-raw">each verdict</span>
        <span>{window}-verdict trailing mean</span>
      </figcaption>
    </figure>
  );
}

function OffPolicy({ result }) {
  if (!result) return null;
  if (result.status !== "ok") {
    const why = {
      "not-enough-data": `${result.n_usable} of 30 recommendations have an outcome`,
      "no-model": "not enough verdicts to fit a model",
      "item-space-changed": "a logged title has left the item space",
      "estimate-unavailable": "the estimator declined",
    }[result.status];
    return (
      <div className="offpolicy">
        <span className="label">Off-policy check</span>
        <p className="offpolicy-why">{why}</p>
        <p className="label">
          Rating a title from the Recommendations tab is what produces one.
        </p>
      </div>
    );
  }
  return (
    <div className="offpolicy">
      <span className="label">Off-policy estimate — weaker evidence, indicative only</span>
      <div className="offpolicy-pair">
        <div>
          <span className="label">Slates actually shown</span>
          <div className="reading num">{(result.logged_value * 10).toFixed(2)}</div>
        </div>
        <div>
          <span className="label">Today&rsquo;s model would have</span>
          <div className="reading num">{(result.estimate * 10).toFixed(2)}</div>
        </div>
      </div>
      <p className="label">over {result.n_usable} logged recommendations you later rated</p>
    </div>
  );
}

export default function Evidence() {
  const { data, error, loading } = useAsync(() => api.audit(), []);
  const validation = useAsync(() => api.validationSummary(), []);

  if (loading) {
    return (
      <div className="page evidence-loading">
        <div className="evidence-working">
          <Organism compact verdicts={200} confidence={0.3} working reaching={0} />
        </div>
        <p className="label">
          refitting the model once per verdict — this is the slow one
        </p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="page">
        <div className="empty">
          <p>{error.message}</p>
          <p>Give it a few more verdicts and come back.</p>
        </div>
      </div>
    );
  }

  const readings = data?.readings ?? [];
  const summary = validation.data;

  return (
    <div className="page">
      <header>
        <span className="label">Evidence</span>
        <h1>Is it learning you?</h1>
        <p>
          Every prediction below was made by a model that had not seen that
          verdict — it walks your history forwards, refitting each step. No
          number here is reported without the interval or the p-value that
          tells you how much to believe it.
        </p>
      </header>

      <Curve curve={data?.curve} />

      <div className="readings">
        {readings.map((r) => {
          // The terminal table has no room for a second line, so a reading
          // carries its unit inside the value — "+0.071 per 100 verdicts".
          // Here there is room, and a 32px unit is not a number.
          //
          // Not every value is a number with a unit though: "0.00 → 1.87" is
          // a pair, and splitting it leaves an arrow stranded under a figure
          // it no longer belongs to.
          const splittable = !r.value.includes("→");
          const [figure, ...unit] = splittable ? r.value.split(" ") : [r.value];
          return (
            <div key={r.measure} className={`reading-cell tone-${r.tone || "none"}`}>
              <span className="label">{r.measure}</span>
              <div className="reading num">{figure}</div>
              {unit.length ? <span className="reading-unit label">{unit.join(" ")}</span> : null}
              {r.reading ? <span className="reading-note label">{r.reading}</span> : null}
            </div>
          );
        })}
      </div>

      <hr className="hairline" />
      <OffPolicy result={data?.off_policy} />

      <hr className="hairline" />
      <h2>Blind test</h2>
      <p className="label">
        predictions frozen to disk before any verdict was known
      </p>
      {summary ? (
        summary.completed_cases === 0 ? (
          <div className="empty">
            <p>No sealed batch has been completed yet.</p>
            <p>
              Seal at least twenty titles you have watched but never rated, then
              reveal them one at a time.
            </p>
          </div>
        ) : (
          <div className="validation">
            <div className="reading-cell">
              <span className="label">Completed cases</span>
              <div className="reading num">{summary.completed_cases}</div>
            </div>
            <div className="reading-cell">
              <span className="label">Top-10 hit rate — full model</span>
              <div className="reading num">{summary.full?.top10_hit_rate?.toFixed(2) ?? "—"}</div>
              <span className="reading-note label">
                {summary.full?.top10_hit_rate_ci95
                  ? `95% ${summary.full.top10_hit_rate_ci95.map((v) => v.toFixed(2)).join(" – ")}`
                  : "interval not available"}
              </span>
            </div>
            <div className="reading-cell">
              <span className="label">Top-10 hit rate — ridge baseline</span>
              <div className="reading num">{summary.ridge?.top10_hit_rate?.toFixed(2) ?? "—"}</div>
            </div>
            <div className="reading-cell">
              <span className="label">Verdict</span>
              <div className="reading-sm num">{summary.decision}</div>
            </div>
          </div>
        )
      ) : null}
    </div>
  );
}
