# Agent design

Decisions for the agent as a whole, phase by phase, in the order they were made. Each section
records the decision, what was rejected, and what is still open. Piece-level detail lives in
the sibling documents (`ACQUISITION.md` for getting a file onto disk).

The method is the `agent-design` skill, which follows *Patterns for Building AI Agents*
(Bhagwat and Gienow, Mastra, 2025). Pattern numbers below refer to that catalogue.

---

## 1. Whiteboard — decided 29 Sep 2026

*Patterns 1 (whiteboard capabilities) and 2 (evolve the architecture).*

Everything the agent could do, grouped by task type:

| Group | Capabilities | State |
| --- | --- | --- |
| Fetch | New releases from TMDB; the back catalogue from the engine's catalogue; Jellyfin library and watch state; free disk | TMDB client exists (`web/live.py`) |
| Judge | Engine score and interval via the `/api/judge` path; a decision rule and threshold; picking the right file; verifying it | File pick and verify built in the spike |
| Act | Acquire from Telegram; file into the library and refresh Jellyfin; evict; notify; leave junk channels | Acquire and file built in the spike |
| Learn | Watched, finished or ignored in Jellyfin becomes a verdict in `events` | Nothing |

Not yet covered, recorded so it is not forgotten: TV series (the spike refuses episodes),
subtitles, notifying the user that a title is ready.

**Decision: the first piece is a manual trigger.** The user asks for a title — a "Get it"
action on any judged title in the web app, and the same verb on the CLI — and the agent
acquires it, files it into Jellyfin and records the outcome.

**Rejected: autonomous acquisition first.** Backlog item 14 records that the engine's
predictions are not yet good enough to act on unsupervised: on the real log it does not beat a
running average (item 10). An autonomous watcher built now would spend disk on weak
predictions. The manual trigger delivers value today, and every request it serves is a
labelled example ("the user wanted this") for the decision rule the autonomous piece will need
later. Autonomy is earned, not assumed (pattern 4).

**Decision: no language model in the loop.** Every step of the manual piece is deterministic
code. The only place a model might earn its keep is parsing bot layouts the spike has not seen,
and that is deferred until one is actually met.

**Decision: runs wherever the app runs.** The home server is the normal host, because Jellyfin
is there, but nothing may be bound to that machine. Every machine-specific value — library
path, Telegram session, Jellyfin URL and key — comes from `.env` through `config`, and the
agent reports a missing one as a domain error with a next step rather than guessing a default
that happens to be right on one host.

Open:

- **One Telegram session, one machine at a time.** A Telethon session used from two machines
  at once can be revoked by Telegram. If the app runs on two machines, only one of them may
  acquire. How that is enforced is a phase 2 question.
- **App and library on different machines.** If the app runs on the Mac but the library is on
  the home server, the file has to get there — a mounted share, or the Mac simply does not
  acquire. Phase 2.

---

## 2. Loop and architecture — decided 29 Sep 2026

*Patterns 2 (evolve the architecture) and 5 (parallelise carefully).*

The loop for the manual piece. No step is model-driven.

| Step | What happens | Kind |
| --- | --- | --- |
| Trigger | "Get it" in the web app, or `ent get <title>` on the CLI | Human |
| Resolve | Title becomes a `Target` through TMDB | Deterministic |
| Queue | A job is recorded, because a download takes about ten minutes and cannot be one HTTP request | Deterministic |
| Work | Search, rank, disk check, download, verify, file, Jellyfin refresh — the spike's `acquire()` | Deterministic |
| Report | Job state and progress reach the UI through the event pattern `build_events.py` already uses for builds | Deterministic |
| Feedback | An `acquired` event is logged; watch-state verdicts are a later piece | Deterministic |

**Decision: one agent, no router, no subagents.** There is one task type. A router earns its
place once there are several (pattern 2).

**Decision: one job at a time.** Parallel downloads would compete for disk headroom and for
Telegram's flood limits, and a queue of one is easy to reason about (pattern 5).

**Decision: the worker runs inside the app process** as a background task. The CLI calls the
same function directly, so the web app and the CLI stay two front ends over one code path, as
`AGENTS.md` requires.

**Decision: only a machine configured to acquire can acquire.** A machine acquires only when
`ENTERTAINER_LIBRARY_DIR` and a Telegram session are both present. Anywhere else, "Get it"
returns a domain error saying this machine cannot acquire and naming what is missing. This
closes both phase 1 open items: the Telegram session lives on one machine, so it is never used
from two at once, and a machine without the library never has to ship files to one that has it.

Rejected:

- **A separate worker process or service.** More moving parts — a second process to start,
  supervise and version — for no benefit at one job at a time.
- **Letting any machine acquire and copy the result to the server.** It needs a share or a
  transfer step, and it puts the Telegram session on every machine.

---

## 3. Autonomy and the human — decided 29 Sep 2026

*Pattern 4 (human-in-the-loop).*

The user starts every job, so the question is where they step in after that.

| Action | Cost of a wrong action | Mode |
| --- | --- | --- |
| Pick the file | Wrong film, wrong dub or a poor copy: disk and an evening. ffprobe catches only a wrong length | **Deferred.** The agent picks and files, and shows what it picked. A "wrong file" action deletes it and tries the next candidate |
| An unclear pick | A guess | **Input mid-run.** The job pauses and shows the top three; the user chooses |
| Post the query in the request group | It goes out publicly under the user's name | Accepted. The route depends on it |
| Join the channels a bot demands | Spam and ad channels on the user's account | Automatic, but only on a bot's demand, every join logged, with a "leave joined channels" cleanup |
| Delete or evict | Loses a file | **Never automatic** in the manual piece |

**Why deferred for the pick:** the user wants the film ready, not a confirmation dialog, and
most picks are unambiguous. Each "wrong file" click is a labelled failure of the matcher,
which is exactly the data phase 7's evals need.

Rejected:

- **Approve every pick before download.** Makes the user the bottleneck on every request for
  the sake of the few that are unclear.
- **Never auto-join.** The spike showed that force-subscribe gates are how these bots work;
  refusing to join means refusing the route.

---

## 4. Decision rule — decided 29 Sep 2026

In the manual piece the user has already decided *whether* to get the film; the rule decides
*which file*. The engine threshold for "is this worth getting at all" belongs to the
autonomous piece and is decided there.

Match scores (`acquire/match.py`) run from 0 (refused) to about 17.5: title similarity up to
10, year +2, resolution up to about +3, HEVC +0.5, original language +2, cam or telesync −5.

| Situation | Action |
| --- | --- |
| Top candidate scores 13 or more, and the leading candidates agree on title, year and language | Auto-pick |
| Best score below 13 — usually a missing year or an unknown language | Pause; the user chooses from the top three |
| Leading candidates disagree on title, year or language | Pause; the user chooses from the top three |
| Several copies of the same film at different quality | Not a conflict; take the highest-ranked without asking |
| Every acceptable copy is over the size cap | Pause and offer the smallest; the user may accept it |
| No answer to a pause within 24 hours | Cancel the job. A missed film costs nothing; the user can ask again |

Budgets per job:

| Budget | Limit |
| --- | --- |
| File size | 3 GB (user's choice) |
| Disk | Free space must exceed the file size plus a 5 GB reserve |
| Time | 45 minutes, then the job fails |
| Force-subscribe join rounds | 3 |
| Candidates tried | 3 |

The 3 GB cap will exclude some 1080p x264 copies of long films; HEVC copies of the same film
usually fit, and the matcher already prefers HEVC slightly.

---

## 5. Context — skipped

No language model is in the loop (phase 1), so there is no context window to engineer.
Patterns 7–9 return if a model is ever added, for example to parse an unfamiliar bot layout.

---

## 6. Failure modes and metrics — decided 29 Sep 2026

*Patterns 10 (failure modes), 11 (business metrics) and 12 (cross-reference).*

Why the agent gets it wrong, by cause:

| Cause | Examples |
| --- | --- |
| Wrong target | TMDB resolves a remake or the wrong year — *Vikram* 1986 against 2022 |
| Bad labels | A dub labelled as the original; no year in the label; a mislabelled file |
| Verification gap | Right length, wrong film or wrong audio. ffprobe cannot tell |
| Source broken | Bot layout changes; no results; a join loop; FloodWait; the file deleted before download; the account restricted |
| Machine | Disk full; library not mounted; Jellyfin unreachable; ffprobe missing |

**North star: wrong-file rate** — "wrong file" clicks divided by files filed. It is the
costliest error because it is silent: the user finds out when they sit down to watch.
**Target: under 5% after 20 jobs.**

Supporting metrics:

| Kind | Metric |
| --- | --- |
| Accuracy | Not-found rate — the film exists on the source but the agent did not find it; pause rate — how often the user is asked to choose |
| Outcome | Success rate; median time from request to ready; joins per job, as a proxy for the Telegram account's health |

Cross-reference: wrong targets and bad labels drive the north star; broken sources drive the
not-found rate. The first eval work therefore goes into matching.

---

## 7. Evals — decided 29 Sep 2026

*Patterns 13 (iterate against evals), 14 (test suite), 15 (expert labels), 16 (datasets from
production).*

- **Capture.** Every job writes its full candidate list, the pick and the outcome — filed,
  "wrong file", or paused and chosen — to `data/acquire/jobs/*.jsonl`. Production outcomes
  become test cases without extra work.
- **Labels come from the user**, who is the domain expert. A "wrong file" click or a choice at
  a pause is the label; labelling costs nothing beyond using the feature.
- **Seed set.** The user names about twenty films weighted towards traps — remakes, films
  sharing a name, dubbed releases, films whose files carry no year. Their real candidate lists
  are recorded once and the user confirms the right file for each.
- **Offline replay.** Matching and ranking are re-run over the recorded lists with no
  Telegram connection. Measured: wrong picks and pauses. **Gate: a change that raises the
  wrong-pick count does not merge.**
- **Privacy.** The real set lives under `data/`, gitignored: the titles a user requests are
  taste data, and this repository is public. The committed tests keep a small synthetic set,
  as `tests/test_acquire.py` does today.
- **No LLM judge.** A pick is right or wrong against a label; there is nothing to grade.

Open: the twenty seed films, from the user.

---

## 8. Security — decided 29 Sep 2026

*Patterns 18 (lethal trifecta), 19 (sandbox), 20 (granular access), 21 (guardrails).*

All three legs of the trifecta are present:

- **Private data:** the Telegram session (full control of an account), `.env` keys, taste data.
- **Untrusted content:** bot messages, file names and the downloaded files themselves.
- **Outbound action:** posting in a request group and joining channels.

No model reads the untrusted content, so prompt injection does not apply. Deterministic code
still acts on what bots send, so it is guarded:

| Guard | Blocks |
| --- | --- |
| Join only `t.me` links, only from the bot that was messaged, only in a message with no file | Bots steering the account into arbitrary channels |
| Video extensions only, and ffprobe must find a video stream | A `.mkv.exe`, an `.apk` or an archive posing as a film |
| Library file names built from TMDB, never from the sender's name | Path traversal and hostile names in the library |
| Nothing is executed or unpacked; downloads stay in `.staging` until verified | Malware reaching the library |
| "Get it" sits behind the web token already required off loopback | Anyone else on the network triggering downloads |
| `--dry-run` as the planning mode: search and rank, no joins, no download | Mistakes while trying a new title or source (pattern 20) |

Pattern 19 (sandboxing) does not apply: the agent executes no code it receives.

**Decision: the Telegram account is configurable, and the personal account is used for now.**
A dedicated account on a second number is the safer choice — a leaked session, a ban, spam
joins and public queries would then stay off the user's own account — but no second number is
available. The account is whichever one `login` signs into; the session file's location comes
from configuration, so switching to a dedicated account later is a fresh login, not a code
change. Until then the mitigations are the join guard above, the join log with its cleanup
action, and the session file readable by its owner only.

Open: regenerate the `api_hash` at my.telegram.org — it was printed into a chat transcript
during the spike.

---

## 9. Operations — decided 29 Sep 2026

| Concern | Decision |
| --- | --- |
| Where | Inside the app process on the home server; `./scripts/entertainer start` brings the worker up with the app |
| Scheduling | None in the manual piece — a queue, one job at a time |
| Login | Headless, over ssh on the server, with `ent acquire login` (the spike's `login`, moved onto the CLI) |
| Logging | Every action with its reason goes to the job's JSONL — search, candidates, the pick and why, joins, download, verification, filing — and the UI shows it the way Build Studio shows a build |
| Stop | A Cancel action in the UI; Ctrl-C on the CLI; `ENTERTAINER_ACQUIRE=off` switches acquisition off without a code change |
| Restart mid-job | The job is marked failed and its `.part` file deleted. No resume: the bot deletes its copy after five minutes, so a resume would fail anyway. The user asks again |
| Undo | "Wrong file" removes the filed copy; "leave joined channels" undoes the joins recorded in the log; removing a film from the library is manual only |

---

## State

**Design settled 29 Sep 2026.** Implementation order is in `docs/BACKLOG.md`, item 14.
