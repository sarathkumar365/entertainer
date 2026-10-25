import { useCallback, useEffect, useState } from "react";

import { api } from "../api";
import Unavailable from "../components/Unavailable";
import { useAsync } from "../useAsync";

/**
 * Films being fetched, and the ones that were.
 *
 * The agent picks a copy on its own when the evidence is clear, so most jobs
 * need nothing from you. It stops and asks when the labels are thin or
 * disagree, and "Wrong file" is the one correction that matters most: each
 * click is a labelled mistake, and the wrong-file rate is what this whole
 * piece is measured by.
 */
const ACTIVE = new Set(["queued", "searching", "downloading"]);
const POLL_MS = 2000;

const gb = (n) => (n ? `${(n / 2 ** 30).toFixed(2)} GB` : "? GB");
const pct = (x) => (x == null ? "—" : `${Math.round(x * 100)}%`);

function Measures({ m }) {
  const done = m.filed >= m.target_jobs;
  return (
    <div className="dl-measures">
      <div>
        <span className="label">Wrong-file rate</span>
        <div className="dl-figure num">{pct(m.wrong_rate)}</div>
        <span className="label">
          target under {pct(m.target_wrong_rate)}
          {done ? "" : ` · ${m.filed} of ${m.target_jobs} filed so far`}
        </span>
      </div>
      <div>
        <span className="label">Asked you</span>
        <div className="dl-figure num">{pct(m.pause_rate)}</div>
        <span className="label">of requests paused for a choice</span>
      </div>
      <div>
        <span className="label">Filed</span>
        <div className="dl-figure num">{m.filed}</div>
        <span className="label">{m.failed} failed · {m.jobs} asked for</span>
      </div>
    </div>
  );
}

function Progress({ p }) {
  if (!p) return null;
  const f = p.total ? p.done / p.total : null;
  return (
    <div className="dl-progress">
      <div className="dl-bar">
        <div style={{ width: f == null ? "100%" : `${(f * 100).toFixed(1)}%` }} />
      </div>
      <span className="label num">
        {gb(p.done)}
        {p.total ? ` of ${gb(p.total)}` : ""}
      </span>
    </div>
  );
}

function Job({ job, onChanged }) {
  const [open, setOpen] = useState(job.state === "paused");
  const [detail, setDetail] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    try {
      setDetail(await api.download(job.id));
    } catch (e) {
      setError(e.message);
    }
  }, [job.id]);

  // A pause is a question for you, so it opens itself.
  useEffect(() => {
    if (job.state === "paused") setOpen(true);
  }, [job.state]);

  useEffect(() => {
    if (open) load();
  }, [open, load, job.state, job.updated]);

  async function act(fn) {
    setBusy(true);
    setError(null);
    try {
      await fn();
      onChanged();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <li className={`dl-job enter dl-${job.state}`}>
      <div className="dl-row">
        <button type="button" className="dl-name" onClick={() => setOpen(!open)}>
          <span className="shelf-title">{job.folder || job.title || "resolving…"}</span>
          <span className="label">{job.via} · {job.created.replace("T", " ").slice(0, 16)}</span>
        </button>
        <span className="label dl-state">{job.state}</span>
        <div className="shelf-actions">
          {job.state === "filed" ? (
            <button type="button" disabled={busy} onClick={() => act(() => api.wrongFile(job.id))}>
              Wrong file
            </button>
          ) : null}
          {ACTIVE.has(job.state) || job.state === "paused" ? (
            <button type="button" disabled={busy} onClick={() => act(() => api.cancelDownload(job.id))}>
              Cancel
            </button>
          ) : null}
        </div>
      </div>
      {job.reason ? <p className="label dl-reason">{job.reason}</p> : null}
      <Progress p={job.progress} />
      {error ? <p className="label dl-error">{error}</p> : null}

      {open && detail ? (
        <div className="dl-detail">
          {job.state === "paused" && detail.offer.length ? (
            <ol className="dl-offer">
              {detail.offer.map((c, i) => (
                <li key={`${c.source}-${c.label}`}>
                  <div>
                    <div className="shelf-title">{c.label}</div>
                    <span className="label num">
                      {gb(c.size)} · {c.source} · score {c.score.toFixed(1)} {c.why}
                    </span>
                  </div>
                  <button
                    type="button"
                    className="cta"
                    disabled={busy}
                    onClick={() => act(() => api.chooseDownload(job.id, i))}
                  >
                    Take this
                  </button>
                </li>
              ))}
            </ol>
          ) : null}
          {detail.filed ? <p className="label">in the library: {detail.filed.path}</p> : null}
          {detail.joins.length ? (
            <p className="label">
              joined for this: {detail.joins.map((j) => `${j.link}${j.left ? " (left)" : ""}`).join(", ")}
            </p>
          ) : null}
          {detail.log.length ? (
            <ol className="dl-log">
              {detail.log.map((line, i) => (
                <li key={i} className="label">
                  <span className="num">{line.at.slice(11, 19)}</span> {line.msg}
                </li>
              ))}
            </ol>
          ) : null}
        </div>
      ) : null}
    </li>
  );
}

export default function Downloads() {
  const { data, error, loading, reload } = useAsync(() => api.downloads(), []);
  const [leaving, setLeaving] = useState(null);
  const live = data?.jobs.some((j) => ACTIVE.has(j.state));

  // Poll only while something is moving; a finished list does not change by itself.
  useEffect(() => {
    if (!live) return undefined;
    const timer = setInterval(reload, POLL_MS);
    return () => clearInterval(timer);
  }, [live, reload]);

  async function leave() {
    setLeaving("leaving…");
    try {
      const r = await api.leaveJoined();
      setLeaving(r.message);
      reload();
    } catch (e) {
      setLeaving(e.message);
    }
  }

  if (error) {
    return (
      <div className="page">
        <Unavailable error={error} />
      </div>
    );
  }

  const joined = data?.joined ?? 0;
  return (
    <div className="page">
      <header>
        <span className="label">Downloads</span>
        <h1>Getting films into the library</h1>
        <p>
          Ask for a film from “Will I like it?” on Home. It picks a copy itself when the
          labels are clear and asks you when they are not. If a filed copy turns out to be
          the wrong film, say so: it is deleted, the next copy is tried, and that is the
          number this is judged by.
        </p>
      </header>

      {loading && !data ? <p className="label">loading…</p> : null}
      {data && !data.ready ? <p className="dl-problem">{data.problem}</p> : null}
      {data ? <Measures m={data.measures} /> : null}
      {data?.jellyfin ? <p className="label dl-jellyfin">{data.jellyfin.message}</p> : null}

      {data ? (
        <section className="shelf">
          <header>
            <span className="label">Jobs</span>
            <span className="label num">{data.jobs.length}</span>
          </header>
          {data.jobs.length === 0 ? (
            <p className="shelf-empty label">nothing asked for yet</p>
          ) : (
            <ul className="dl-list">
              {data.jobs.map((job) => (
                <Job key={job.id} job={job} onChanged={reload} />
              ))}
            </ul>
          )}
        </section>
      ) : null}

      {data ? (
        <section className="shelf">
          <header>
            <span className="label">Joined channels</span>
            <span className="label num">{joined}</span>
          </header>
          <p className="label shelf-note">
            Bots demand that the account join their channels before they send a file. Each join
            is recorded against the job that caused it.
          </p>
          <div className="shelf-actions">
            <button type="button" disabled={!joined || leaving === "leaving…"} onClick={leave}>
              Leave joined channels
            </button>
          </div>
          {leaving ? <p className="label">{leaving}</p> : null}
        </section>
      ) : null}
    </div>
  );
}
