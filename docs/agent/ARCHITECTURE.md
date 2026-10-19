# Agent architecture

How the manual piece is built: where the code lives, which process and thread run it, who
owns the Telegram session, and how its outcome reaches the engine. `DESIGN.md` holds what the
agent does and why; this file holds how it is structured. Decided 29 Sep 2026.

The spike (`836ef8c`) proved the route but is one function, `acquire()`, that searches, ranks,
downloads and files in a single call. Everything below is what has to change so it can pause,
resume, run behind the web app and be measured.

---

## Structure

```
acquire/                  capability — sources, matching, verification. Imports config and errors only
  match.py                Target, Candidate, score()
  verify.py               ffprobe checks (moved out of core.py)
  sources.py              Source Protocol and the registry of named sources
  telegram.py archive.py  the sources
agent/                    work layer — orchestration
  decide.py               the phase 4 decision rule
  jobs.py                 the job state machine and its JSONL record
  worker.py               the thread that runs jobs, one at a time
  library.py              filing into the library and the Jellyfin refresh
commands/acquire.py       ent acquire login | get [--dry-run] | replay
web/routers/downloads.py  get, list jobs, job stream, choose, cancel, wrong file
config.py                 AcquireSettings — the only place the agent's environment is read
```

**Decision: `acquire/` is a capability, `agent/` is the work.** A capability package that
imported `store` to log an event would break the dependency rule in `AGENTS.md`.

**Rejected: everything in `acquire/`.** Fewer packages, but it puts orchestration and
persistence under the capability layer.

The web router is named `downloads`, not `library`: `web/routers/library.py` is already the
user's watched-and-watchlist shelf, and the Jellyfin library is a different thing.

## Contracts

- **`Source` Protocol.** `name`, async context manager, `search(target) -> list[Candidate]`,
  `fetch(candidate, dest, progress) -> Path`. Sources are chosen by name from `AcquireSettings`.
  One source failing in `search` is logged and skipped, never fatal to the job.
- **`Target` comes from the IMDb id.** "Get it" acts on a judged title, which already carries
  an `imdb_id`; the target is built from TMDB's find-by-external-id, not a title search. This
  removes the wrong-target failure mode (*Vikram* 1986 against 2022). The CLI resolves typed text
  through `resolve.py` to an item first, and so inherits its disambiguation.
- **`plan()` and `execute()`.** `plan(target)` searches, ranks and returns a `Decision` from
  `decide.py`: `AutoPick`, `Pause(top3, reason)` or `OverCap(smallest)`. `execute(target,
  candidate)` fetches, verifies and files. The job sits between the two, which is what makes a
  pause, a choice and "wrong file, try the next" possible.
- **Candidates go stale.** A bot's `file#<id>` callback does not survive its five-minute
  deletion window, so resuming after a pause or a "wrong file" plans again rather than reusing
  the stored list. The stored list is kept as eval data only.

## Process and threads

**Decision: one worker thread with its own asyncio loop,** started in the app's lifespan and
stopped on shutdown. Web handlers are synchronous and Telethon is async; the worker also
blocks on ffprobe, a synchronous Jellyfin call and file moves. On uvicorn's own loop, one
missed blocking call freezes the UI.

- The Telethon client is created inside the worker thread, on that thread's loop.
- DuckDB writes from the worker use a cursor on the app's connection, never a second
  connection.
- Staging is `<library>/.staging`, on the library's filesystem, so filing is a rename, not a
  3 GB copy.

**Rejected:** a task on uvicorn's loop (fragile); a separate process (rejected in `DESIGN.md`
phase 2).

## The Telegram session

**Decision: a lock file, `data/telegram/acquire.lock`, held only while a job runs.** The client
connects per job, as the spike already does. Whoever holds the lock acquires; anyone else gets a
domain error — "the app is acquiring; wait, or use the web page". `ent acquire login` takes the
same lock.

Two processes opening one Telethon session file collide on its SQLite database at best and get
the session revoked at worst.

**Rejected:** the CLI handing jobs to the running app over HTTP (two code paths, and the CLI
depends on the app); the app always owning the session (the CLI cannot work without the app).

## Jobs

One JSONL file per job under `data/acquire/jobs/`, append-only, the same files-as-boundary
pattern as `build_events.py`. The first line is the request; every later line is an action with
its reason.

```
queued → searching → downloading → verifying → filed
             ↓            ↑    ↓          ↓
           paused ── chosen    failed   failed / wrong → searching
             ↓
         cancelled (user, or 24 h without an answer)
```

- One job at a time; the rest wait in `queued`.
- 45 minutes per job, then `failed`.
- On app start, any job not in a terminal state is marked `failed` and its `.part` deleted.
  No resume.
- `ENTERTAINER_ACQUIRE=off` refuses new jobs and leaves existing records readable.

## The engine

**Decision: a request is a verdict.** Each job logs `kind='request'`, `source='agent'` to
`events`. `request` joins `CONSUMED_KINDS`, so a requested title stops being recommended.

Every reader of `events` filters by `kind`, so the new kind cannot leak into training, and the
profile export keys rows by IMDb id, so the verdict survives a rebuild. It is also the label
the autonomous decision rule will need.

**Rejected:** keeping requests only in the job files. The engine would never learn that the
user wanted the title.

---

## Phases

Each phase merges on its own; phase 4 needs phase 3.

| # | Phase | Done when |
| --- | --- | --- |
| 1 | **Contracts.** `AcquireSettings`; `Source` Protocol and registry; `Target` from IMDb id; `plan()` / `execute()` split; the four review findings from `836ef8c` fixed (staging location, one oversized candidate aborting the job, one source error killing the run, the guessed `.mkv` extension); `commands/acquire.py` replaces `python -m entertainer.acquire`; the session lock | `ent acquire get` works end to end; each fixed bug has a test; ruff clean |
| 2 | **Decision rule.** `agent/decide.py`; a pause on the CLI becomes a prompt | The `DESIGN.md` phase 4 table covered by synthetic tests |
| 3 | **Jobs and worker.** State machine, JSONL, worker thread, restart cleanup, pause and time budgets, the `request` event | A job killed mid-run returns as failed; a requested title leaves the recommendations |
| 4 | **Web.** `downloads` router, a Downloads page, "Get it" on judged titles, behind the existing token | Verified in the browser: get, pause and choose, wrong file, cancel |
| 5 | **Join log and cleanup.** Every join recorded against its job; "leave joined channels" | A join appears in the log and the cleanup leaves it |
| 6 | **Eval replay.** `ent acquire replay` over `data/acquire/jobs/`: wrong picks and pauses | Runs over the seed set; the number becomes the merge gate for matcher changes |
| 7 | **Home server.** Jellyfin refresh against the real server; the north star tracked | Twenty jobs, wrong-file rate under 5% |

## Not in this plan

The autonomous watcher and its engine threshold, disk eviction, TV series, subtitles,
watch-state verdicts from Jellyfin, and any language model.

## Risks

| Risk | Signal | Mitigation |
| --- | --- | --- |
| Telethon client bound to the wrong loop | "attached to a different loop" errors | Create the client inside the worker thread |
| DuckDB written from two places | Lock errors | Worker cursor on the app's connection |
| Filing copies across filesystems | A multi-minute pause after verify | Staging inside the library directory |
| A bot changes its layout | Not-found rate jumps in the job log | Watch the rate; the parser is the one place a model might later earn its keep |

## State

**Phase 1 done 30 Sep 2026.** `ent acquire get --imdb tt0013442 --sources archive` searched,
downloaded, verified (93 min against TMDB's 94) and filed *Nosferatu (1922)* into a scratch
library, with staging inside it. Found on the way: the catalogue resolver drops a typed year
it cannot match and returns a same-named film from another decade as confident, so
`ent acquire get` refuses a match whose year contradicts the typed one; and film extras
("Behind the Scenes") scored as the film, so the matcher now refuses them. The Telegram
sources were not run live in this phase.
