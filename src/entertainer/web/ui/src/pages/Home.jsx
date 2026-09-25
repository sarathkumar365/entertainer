import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";

import { api } from "../api";
import Organism from "../components/Organism";
import { settledness } from "../components/TasteField";
import { useAsync } from "../useAsync";

// Long enough that a fast answer still reads as considered rather than
// flashing past; short enough not to feel staged.
const THINK_AT_LEAST = 900;
const COUNT_FOR = 700;

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const reduced = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/** Catalogue hits first, since those are judged on everything it knows. */
function mergeMatches(found) {
  const known = (found.catalogue ?? []).map((item) => ({ ...item, known: true }));
  const fresh = (found.tmdb ?? []).map((item) => ({ ...item, known: false }));
  return [...known, ...fresh].slice(0, 5);
}

function judgeBody(match) {
  // A catalogue title sends its TMDB id too, in case the item space lacks it.
  const tmdb = match.tmdb_id ? { tmdb_id: match.tmdb_id, kind: match.kind } : {};
  return match.known ? { item_id: match.item_id, ...tmdb } : tmdb;
}

function useCountUp(target) {
  const [value, setValue] = useState(0);
  useEffect(() => {
    if (target == null) return undefined;
    if (reduced()) {
      setValue(target);
      return undefined;
    }
    let frame;
    const start = performance.now();
    const tick = (now) => {
      const f = Math.min(1, (now - start) / COUNT_FOR);
      setValue(target * (1 - Math.pow(1 - f, 3)));
      if (f < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [target]);
  return value;
}

/**
 * The model, and three numbers. No cards, no grid.
 *
 * Confidence is derived from how much it has been told rather than asked for
 * separately: the audit endpoint that would give a real figure costs two
 * seconds, and the home screen should not.
 *
 * "Will I like it?" happens here too, in the organism itself: it flattens
 * into the search box, churns while it thinks, and answers with its body.
 */
export default function Home() {
  const { data: progress } = useAsync(() => api.progress(), []);
  const { data: mode } = useAsync(() => api.mode(), []);

  // idle -> ask -> (choose) -> thinking -> result | failed
  const [phase, setPhase] = useState("idle");
  const [query, setQuery] = useState("");
  const [choices, setChoices] = useState([]);
  const [active, setActive] = useState(0);
  const [answer, setAnswer] = useState(null);
  const [failure, setFailure] = useState(null);
  const [searching, setSearching] = useState(false);
  const inputRef = useRef(null);
  // Bumped whenever the question changes, so a late answer to an abandoned
  // one is dropped instead of landing on the wrong screen.
  const asking = useRef(0);

  const verdicts = progress?.rated ?? 0;
  // Derived rather than measured; the Evidence tab is where the actual
  // number lives. Same curve as the Taste figure, though that one is fed the
  // model's effective count, which can differ from this raw one.
  const confidence = settledness(verdicts);

  // The reaching lines are the last few verdicts, and the lit ends are the
  // ones you liked. Not decoration: with an empty log there is nothing
  // reaching, which is the truthful picture of a model that knows nothing.
  const recent = progress?.recent ?? [];
  const enjoyed = recent.filter((r) => r.verdict === "love" || r.verdict === "like").length;

  const prediction = phase === "result" ? answer?.prediction : null;
  const shown = useCountUp(prediction ? prediction.score : null);

  function open() {
    asking.current += 1;
    setPhase("ask");
    setAnswer(null);
    setFailure(null);
    setChoices([]);
    setQuery("");
    setSearching(false);
  }

  function close() {
    asking.current += 1;
    setPhase("idle");
    setAnswer(null);
    setFailure(null);
    setChoices([]);
    setSearching(false);
  }

  useEffect(() => {
    if (phase === "ask" || phase === "choose") {
      // After the membrane has mostly flattened, so the caret does not
      // appear in mid-air.
      const timer = setTimeout(() => inputRef.current?.focus(), reduced() ? 0 : 320);
      return () => clearTimeout(timer);
    }
    return undefined;
  }, [phase]);

  useEffect(() => {
    if (phase === "idle") return undefined;
    const onKey = (event) => {
      if (event.key === "Escape") close();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [phase]);

  async function judge(match) {
    const ticket = ++asking.current;
    setChoices([]);
    setPhase("thinking");
    setAnswer({ item: match });
    try {
      const [judged] = await Promise.all([api.judge(judgeBody(match)), wait(THINK_AT_LEAST)]);
      if (ticket !== asking.current) return;
      setAnswer(judged);
      setPhase("result");
    } catch (error) {
      if (ticket !== asking.current) return;
      setFailure(error);
      setPhase("failed");
    }
  }

  async function submit(event) {
    event?.preventDefault();
    const q = query.trim();
    if (!q || searching) return;
    if (phase === "choose" && choices[active]) {
      judge(choices[active]);
      return;
    }
    const ticket = ++asking.current;
    setSearching(true);
    setFailure(null);
    try {
      const matches = mergeMatches(await api.search(q));
      if (ticket !== asking.current) return;
      if (matches.length === 1) {
        // Before judge() takes a new ticket, or the finally below would
        // leave the box refusing every later search.
        setSearching(false);
        judge(matches[0]);
      } else {
        setChoices(matches);
        setActive(0);
        setPhase("choose");
      }
    } catch (error) {
      if (ticket !== asking.current) return;
      setFailure(error);
    } finally {
      if (ticket === asking.current) setSearching(false);
    }
  }

  function onInputKey(event) {
    if (phase !== "choose" || choices.length === 0) return;
    if (event.key === "ArrowDown" || event.key === "ArrowRight") {
      event.preventDefault();
      setActive((i) => (i + 1) % choices.length);
    } else if (event.key === "ArrowUp" || event.key === "ArrowLeft") {
      event.preventDefault();
      setActive((i) => (i - 1 + choices.length) % choices.length);
    }
  }

  const asked = phase !== "idle";
  const searchOpen = phase === "ask" || phase === "choose";
  const item = answer?.item;
  const early = failure?.code === "not_enough_evidence";

  return (
    <div className={`home${asked ? " home-asked" : ""}`}>
      <Organism
        verdicts={verdicts}
        confidence={confidence}
        reaching={recent.length}
        rated={enjoyed}
        shape={searchOpen ? "pill" : "blob"}
        thinking={phase === "thinking" || searching}
        result={
          prediction
            ? {
                score: prediction.score,
                low: prediction.interval_low,
                high: prediction.interval_high,
                probability: prediction.like_probability,
              }
            : null
        }
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

      <form className={`home-ask${searchOpen ? " on" : ""}`} onSubmit={submit} role="search">
        <input
          ref={inputRef}
          value={query}
          onChange={(event) => {
            setQuery(event.target.value);
            if (phase === "choose") setPhase("ask");
            setChoices([]);
          }}
          onKeyDown={onInputKey}
          placeholder="a title — a new release works too"
          aria-label="Title to ask about"
          tabIndex={searchOpen ? 0 : -1}
          spellCheck={false}
          autoComplete="off"
        />
        <button className="home-ask-go" type="submit" aria-label="Will I like it?" tabIndex={searchOpen ? 0 : -1}>
          <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
            <circle cx="12" cy="12" r="8.5" fill="none" stroke="currentColor" strokeWidth="1.2" />
            <circle cx="12" cy="12" r="2.2" fill="currentColor" />
          </svg>
        </button>
      </form>

      {phase === "choose" ? (
        <div className="home-choices" role="listbox" aria-label="Which one?">
          {choices.length === 0 ? (
            <p className="home-line">nothing by that name — try the original title</p>
          ) : (
            <>
              <p className="label home-choices-label">which one?</p>
              {choices.map((match, i) => (
                <button
                  key={match.known ? `c${match.item_id}` : `t${match.tmdb_id}`}
                  type="button"
                  role="option"
                  aria-selected={i === active}
                  className={`home-choice enter${i === active ? " on" : ""}`}
                  style={{ animationDelay: `${i * 45}ms` }}
                  onMouseEnter={() => setActive(i)}
                  onClick={() => judge(match)}
                >
                  {match.poster ? <img src={match.poster} alt="" loading="lazy" /> : <span className="home-choice-blank" />}
                  <span className="home-choice-text">
                    <span className="home-choice-title">{match.title}</span>
                    <span className="home-choice-meta num">
                      {[match.year, match.kind === "tv" ? "series" : "film", match.language_name]
                        .filter(Boolean)
                        .join(" · ")}
                    </span>
                  </span>
                  <span className="home-choice-tag label">{match.known ? "in catalogue" : "new"}</span>
                </button>
              ))}
            </>
          )}
        </div>
      ) : null}

      {phase === "result" && prediction ? (
        <div className="home-score" aria-live="polite">
          <span className="num">{shown.toFixed(1)}</span>
          <small className="num">/10</small>
        </div>
      ) : null}

      {phase === "idle" ? (
        <div className="home-centre enter">
          <h1>entertainer</h1>
          <p>
            It learns what you like from titles and verdicts alone — no genres given
            to it, no tags. Feed it and watch it tighten.
          </p>
          <div className="home-actions">
            <Link className="cta" to="/recs">See what it thinks</Link>
            <button type="button" className="cta cta-quiet" onClick={open}>Will I like it?</button>
            <Link className="cta cta-quiet" to="/rate">Rate something</Link>
          </div>
        </div>
      ) : (
        <div className="home-centre home-answer" key={phase}>
          {phase === "ask" ? (
            failure ? (
              <p className="home-line enter">{failure.message}</p>
            ) : (
              <p className="home-line enter">
                name anything — it will say how close it sits to your taste
              </p>
            )
          ) : null}

          {phase === "thinking" ? (
            <p className="home-line enter">
              reading <em>{item?.title}</em>…
            </p>
          ) : null}

          {phase === "result" && prediction ? (
            <div className="enter">
              <h2 className="home-verdict-title">
                {item.title}
                {item.year ? <span className="num"> · {item.year}</span> : null}
              </h2>
              <p className={`home-verdict${prediction.likely_like ? " liked" : ""}`}>
                {prediction.likely_like ? "Likely to like it" : "Probably not for you"}
              </p>
              <p className="home-line num">
                {Math.round(prediction.like_probability * 100)}% chance you like it · 90% range{" "}
                {prediction.interval_low.toFixed(1)}–{prediction.interval_high.toFixed(1)}
              </p>
              {answer.source === "text" ? (
                <p className="home-line home-note">new to it — judged on its description and TMDB rating</p>
              ) : null}
            </div>
          ) : null}

          {phase === "failed" ? (
            <div className="enter">
              {early ? (
                <p className="home-line">
                  Rate a few titles first — after three verdicts it can guess.{" "}
                  <Link to="/rate">Rate something</Link>
                </p>
              ) : (
                <p className="home-line">{failure?.message || "That did not work."}</p>
              )}
            </div>
          ) : null}

          {phase === "result" || phase === "failed" ? (
            <div className="home-actions enter">
              <button type="button" className="cta" onClick={open}>{phase === "failed" ? "Try another" : "Ask about another"}</button>
              <button type="button" className="cta cta-quiet" onClick={close}>Done</button>
            </div>
          ) : null}
        </div>
      )}
    </div>
  );
}
