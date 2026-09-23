import { Link } from "react-router-dom";

import { api } from "../api";
import Organism from "../components/Organism";
import { useAsync } from "../useAsync";

/**
 * The model, and three numbers. No cards, no grid.
 *
 * Confidence is derived from how much it has been told rather than asked for
 * separately: the audit endpoint that would give a real figure costs two
 * seconds, and the home screen should not.
 */
export default function Home() {
  const { data: progress } = useAsync(() => api.progress(), []);
  const { data: mode } = useAsync(() => api.mode(), []);

  const verdicts = progress?.rated ?? 0;
  // Derived rather than measured. The audit endpoint reports real interval
  // calibration, but it refits once per verdict and takes seconds — too much
  // for a landing screen. This stands in for it, and the Evidence tab is
  // where the actual number lives.
  const confidence = Math.min(0.95, 0.25 + verdicts / 400);

  // The reaching lines are the last few verdicts, and the lit ends are the
  // ones you liked. Not decoration: with an empty log there is nothing
  // reaching, which is the truthful picture of a model that knows nothing.
  const recent = progress?.recent ?? [];
  const enjoyed = recent.filter((r) => r.verdict === "love" || r.verdict === "like").length;

  return (
    <div className="home">
      <Organism
        verdicts={verdicts}
        confidence={confidence}
        reaching={recent.length}
        rated={enjoyed}
      />

      <div className="home-hud">
        <div className="home-corner home-tl">
          <span className="label">Your taste</span>
          <div className="reading-sm num">{verdicts.toLocaleString()} verdicts</div>
        </div>
        <div className="home-corner home-tr">
          <span className="label">Catalogue</span>
          <div className="reading-sm num">{(mode?.catalogue ?? 0).toLocaleString()}</div>
        </div>
        <div className="home-corner home-bl">
          <span className="label">Languages rated</span>
          <div className="reading-sm num">{progress?.by_language?.length ?? 0}</div>
        </div>
        <div className="home-corner home-br">
          <span className="label">Events</span>
          <div className="reading-sm num">{(progress?.events ?? 0).toLocaleString()}</div>
        </div>
      </div>

      <div className="home-centre enter">
        <h1>entertainer</h1>
        <p>
          It learns what you like from titles and verdicts alone — no genres given
          to it, no tags. Feed it and watch it tighten.
        </p>
        <div className="home-actions">
          <Link className="cta" to="/recs">See what it thinks</Link>
          <Link className="cta cta-quiet" to="/rate">Rate something</Link>
        </div>
      </div>
    </div>
  );
}
