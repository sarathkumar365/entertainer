import { useCallback, useState } from "react";

import { VERDICTS, api } from "../api";
import Organism from "../components/Organism";
import TitleCard from "../components/TitleCard";
import Unavailable from "../components/Unavailable";
import { useAsync } from "../useAsync";

/** Recommendations, and the only place a verdict closes the off-policy loop. */
const OUTCOME = VERDICTS.filter((v) => v.key !== "unseen").concat([
  { key: "unseen", label: "Not yet" },
]);

/** Why a verdict given here is worth more than the same verdict elsewhere.
 *
 * Every slate logs the probability each title had of being shown, so a verdict
 * given from this page can be matched back to the slate that produced it. That
 * is the only thing that can answer "were those picks any good?" — a rating
 * given on the Rate page teaches the model just as much but measures nothing.
 *
 * The README has always said so. No screen did, which is why the off-policy
 * check sat at 16 of the 30 outcomes it needs while hundreds of verdicts went
 * in elsewhere.
 */
function Worth({ outcomes }) {
  if (!outcomes) return null;
  const { have, need } = outcomes;
  const done = have >= need;
  const left = need - have;
  return (
    <p className={`recs-worth label${done ? " recs-worth-done" : ""}`}>
      <strong>Rating here counts twice.</strong>{" "}
      {done ? (
        <>
          Every verdict on this page also measures whether the picks were good,
          and there are now enough of them — see Evidence.
        </>
      ) : (
        <>
          A verdict here also measures whether the picks were good, which no
          other screen can do. {have} of {need} recorded; {left} more
          {left === 1 ? " unlocks" : " unlock"} the check on Evidence.
        </>
      )}
    </p>
  );
}

export default function Recommendations() {
  const [nonce, setNonce] = useState(0);
  const [kind, setKind] = useState("both");
  const { data, error, loading } = useAsync(() => api.slate(9, kind), [nonce, kind]);
  const [answered, setAnswered] = useState({});
  const [pulse, setPulse] = useState(0);

  const give = useCallback(
    async (item, verdict) => {
      setAnswered((prev) => ({ ...prev, [item.item_id]: verdict }));
      setPulse((p) => p + 1);
      try {
        await api.rate({
          item_id: item.item_id,
          verdict,
          // This is what makes the verdict an off-policy datapoint rather
          // than just another rating: it says which slate produced it.
          slate_id: data?.slate_id,
          position: data?.items?.findIndex((i) => i.item_id === item.item_id),
        });
      } catch {
        setAnswered((prev) => {
          const next = { ...prev };
          delete next[item.item_id];
          return next;
        });
      }
    },
    [data],
  );

  // Saving is not a verdict, so it does not go through `give` — it removes
  // the card and keeps the title out of future slates without teaching the
  // model anything.
  const save = useCallback(async (item) => {
    setAnswered((prev) => ({ ...prev, [item.item_id]: "saved" }));
    try {
      await api.libraryAction(item.item_id, "save");
    } catch {
      setAnswered((prev) => {
        const next = { ...prev };
        delete next[item.item_id];
        return next;
      });
    }
  }, []);

  const items = (data?.items ?? []).filter((i) => !(i.item_id in answered));
  const given = Object.keys(answered).length;

  return (
    <div className="page">
      <header className="recs-head">
        <div className="recs-organism">
          <Organism
            compact
            verdicts={40}
            confidence={loading ? 0.25 : 0.8}
            working={loading}
            reaching={0}
            pulseKey={pulse}
          />
        </div>
        <div>
          <span className="label">Recommendations</span>
          <h1>{loading ? "Thinking" : "What to watch next"}</h1>
          <p>
            Each prediction is a distribution, not a number — the wider the curve,
            the less it knows. Saying what happened is the only thing that tells it
            whether a slate was any good.
          </p>
          <Worth outcomes={data?.outcomes} />
          <div className="mode-switch recs-kind">
            {[["both", "Everything"], ["movie", "Films"], ["tv", "Series"]].map(([value, label]) => (
              <button
                key={value}
                type="button"
                className={kind === value ? "on" : ""}
                onClick={() => { setAnswered({}); setKind(value); }}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      </header>

      {error ? <Unavailable error={error} /> : null}

      {!error && !loading && items.length === 0 ? (
        <div className="empty">
          <p>{given ? "That is the whole slate answered." : "Nothing came back."}</p>
          <button type="button" className="cta" onClick={() => { setAnswered({}); setNonce((n) => n + 1); }}>
            New slate
          </button>
        </div>
      ) : null}

      <div className="grid">
        {items.map((item, i) => (
          <div key={item.item_id} className="enter" style={{ animationDelay: `${i * 45}ms` }}>
            <TitleCard
              item={item}
              verdicts={OUTCOME}
              showPrediction
              onVerdict={give}
              onSave={() => save(item)}
            />
          </div>
        ))}
      </div>

      {given > 0 ? (
        <p className="recs-progress label">
          {given} outcome{given === 1 ? "" : "s"} recorded from this slate
        </p>
      ) : null}
    </div>
  );
}
