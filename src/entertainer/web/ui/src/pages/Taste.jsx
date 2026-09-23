import { useMemo, useState } from "react";

import { api } from "../api";
import { useAsync } from "../useAsync";

/**
 * Where your taste sits among everything the catalogue knows.
 *
 * The field is the one place in the product this appears. It is a map, and a
 * map on every page is wallpaper.
 *
 * The points are laid out by a seeded hash rather than by a real projection:
 * the engine has no 2-D embedding, and inventing one that looked meaningful
 * would be a lie. This is scale and density, not coordinates — the honest
 * reading is "the catalogue is vast and you occupy a corner of it".
 */
function LatentField({ axes }) {
  // A wide viewBox with the default preserveAspectRatio, so a circle stays a
  // circle. Stretching a square one to the container turned every point into
  // a dash.
  const W = 320;
  const H = 120;

  const points = useMemo(() => {
    const out = [];
    let seed = 7;
    const rand = () => {
      seed = (seed * 1664525 + 1013904223) % 4294967296;
      return seed / 4294967296;
    };
    for (let i = 0; i < 460; i += 1) {
      const near = i > 340;
      const angle = rand() * Math.PI * 2;
      const radius = near ? rand() * 0.16 : 0.18 + rand() * 0.82;
      out.push({
        x: (near ? W * 0.66 : W / 2) + Math.cos(angle) * radius * (near ? 26 : W * 0.46),
        y: H / 2 + Math.sin(angle) * radius * (near ? 26 : H * 0.44),
        r: near ? 1.0 : 0.45 + rand() * 0.5,
        near,
      });
    }
    return out;
  }, []);

  return (
    <figure className="field">
      <svg viewBox={`0 0 ${W} ${H}`} role="img"
           aria-label="Catalogue titles scattered in the latent space, with the region closest to your taste marked">
        {points.map((p, i) => (
          <circle key={i} cx={p.x.toFixed(1)} cy={p.y.toFixed(1)} r={p.r.toFixed(2)}
                  className={p.near ? "field-near" : "field-far"} />
        ))}
        <circle cx={W * 0.66} cy={H / 2} r="30" className="field-ring" />
        <circle cx={W * 0.66} cy={H / 2} r="46" className="field-ring field-ring-wide" />
        <line x1={W * 0.12} y1={H * 0.84} x2={W * 0.66 - 30} y2={H / 2 + 12}
              className="field-vector" />
        <circle cx={W * 0.12} cy={H * 0.84} r="2" className="field-origin" />
      </svg>
      <figcaption className="label">
        <span>cold start</span>
        <span>you, across {axes} described axes</span>
      </figcaption>
    </figure>
  );
}

function Axis({ axis }) {
  const width = Math.min(100, Math.abs(axis.strength) * 400);
  return (
    <section className="axis enter">
      <header>
        <span className="label">Axis {axis.index}</span>
        <span className="label num">influence {axis.strength.toFixed(3)}</span>
      </header>
      <div className="meter"><i style={{ width: `${width}%` }} /></div>
      <div className="poles">
        <div>
          <div className="label pole-towards">Towards</div>
          <p className="pole-terms">{axis.towards.terms.join(" · ") || "—"}</p>
          <p className="pole-examples">{axis.towards.examples.join(", ")}</p>
        </div>
        <div>
          <div className="label">Away from</div>
          <p className="pole-terms">{axis.away.terms.join(" · ") || "—"}</p>
          <p className="pole-examples">{axis.away.examples.join(", ")}</p>
        </div>
      </div>
    </section>
  );
}

export default function Taste() {
  const [axes] = useState(6);
  const { data, error, loading } = useAsync(() => api.taste(axes), [axes]);

  if (error) {
    return (
      <div className="page">
        <div className="empty">
          <p>{error.message}</p>
        </div>
      </div>
    );
  }

  const side = data?.side_features ?? [];
  const widest = Math.max(0.001, ...side.map((f) => Math.abs(f.weight)));

  return (
    <div className="page">
      <header>
        <span className="label">Taste</span>
        <h1>What it has worked out</h1>
        <p>
          None of this was given to it. The axes are directions in a learned
          space, and they only mean anything relative to what they point away
          from — which is why both poles are shown.
        </p>
      </header>

      <LatentField axes={data?.axes?.length ?? 0} />

      {loading ? <p className="label">reading the model…</p> : null}

      {data ? (
        <>
          <p className="taste-meta label">
            learned from <span className="num">{data.n_verdicts}</span> verdicts ·
            capacity {data.capacity.rff ? `linear + ${data.capacity.rff} RFF` : "linear"} ·
            log Z <span className="num">{data.capacity.log_evidence.toFixed(1)}</span>
          </p>

          {data.axes.length === 0 ? (
            <div className="empty">Not enough signal to describe the axes yet.</div>
          ) : (
            data.axes.map((axis) => <Axis key={axis.index} axis={axis} />)
          )}

          <hr className="hairline" />
          <h2>Surface preferences</h2>
          <p className="label">learned, not assumed</p>
          <div className="surface">
            {side.map((feature) => (
              <div key={feature.name} className="surface-row">
                <span className="surface-name">{feature.name}</span>
                <div className="surface-bar">
                  <i
                    className={feature.weight >= 0 ? "pos" : "neg"}
                    style={{ width: `${(Math.abs(feature.weight) / widest) * 100}%` }}
                  />
                </div>
                <span className="num surface-weight">
                  {feature.weight >= 0 ? "+" : ""}{feature.weight.toFixed(3)}
                </span>
              </div>
            ))}
          </div>
        </>
      ) : null}
    </div>
  );
}
