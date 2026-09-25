# Code health — audit and cleanup plan

Written 25 September 2026, after a full read of the repository against the question the
project has never asked itself: *is this laid out the way a stranger would expect?*

The project was scaffolded and built fast, feature by feature, and nothing has been
reorganised since. That shows up in exactly one place — the package root — and almost
nowhere else. This document records what the audit found, what is already fine and should
be left alone, and the ordered work for turning the findings into enforced rules rather
than good intentions.

Every claim below was produced by running a tool against the tree, not by impression.

> **Second pass (same day).** A behavioural read of the code found problems the tool pass
> could not see, including a personal data file committed to the public repository,
> verdicts reattached to the wrong film on rebuild, and a web app that errors under
> concurrent requests. See [Second pass](#second-pass--what-the-first-audit-missed) at the
> end. It also corrects several claims below; those corrections take precedence.

---

## The short version

| Area | State |
| --- | --- |
| Package layout below the root | Good — seven cohesive subpackages, sane layering |
| Package root | **The problem.** Eighteen flat modules mixing four different layers |
| Frontend | Organised, but has no tooling at all — no linter, no formatter, no tests |
| Dead code | Near zero. Eight real candidates, all small |
| Dependencies | Five declared and unused; one used and undeclared |
| Secrets and git hygiene | Clean. Nothing leaked, `.gitignore` is careful and commented |
| Known CVEs | None (`pip-audit` over the whole installed set) |
| Web auth | Thought through. Two small hardening fixes |
| Enforcement | **None.** `ruff` is configured but nothing runs it; no CI, no pre-commit |

The headline: this codebase is in better shape than "scaffolded fast" implies. The work is
not a rescue. It is (a) tidying one drawer, (b) buying tools so the tidiness survives, and
(c) writing the conventions down so they stop being re-derived from scratch each session.

---

## 1. Structure

### What is already right

`src/entertainer/` has seven subpackages, and each one is a real seam:

```
data/        ingest and enrichment — imdb, netflix, tmdb, catalog, download
models/      the learning — cf, encoder, features, fusion, taste, discover, population, itemcard
evaluation/  the measuring — simulate, prequential, personal, offpolicy, metrics, integrity
coldstart/   elicitation — elicit, session
render/      terminal presentation — tables, theme
web/         the HTTP layer — app, routers/, plus the React app in ui/
commands/    the CLI verbs — build, browse, diagnose, release, transfer, verdicts, serve
```

The dependency direction was measured per package. It mostly runs the right way: `commands`
and `web` sit on top and import downward; `models`, `coldstart` and `data` import almost
nothing but `config` and `errors`. That is the shape you want, and it happened without anyone
enforcing it.

### What is wrong: the root is a drawer, not a layer

Eighteen modules sit loose at the package root, and they belong to four different layers:

| Layer | Modules |
| --- | --- |
| Infrastructure | `config`, `errors`, `resources`, `store` |
| Domain / serving | `engine`, `recommend`, `resolve` |
| Build pipeline | `pipeline`, `ingest`, `build_events`, `manifests` |
| Distribution | `bundle`, `releases`, `archives`, `profile_io`, `setup_status` |
| Entry point | `cli` |

Nothing here is *wrong* in isolation — the modules are individually cohesive. The cost is
navigational: a reader arriving at the root cannot tell the four-times-a-second hot path
(`engine`, `recommend`, `store`) from the once-a-week machinery (`releases`, `archives`,
`bundle`). The subpackages already say *what kind of thing is this*; the root refuses to.

Proposed regrouping, following the layers above:

```
core/        config, errors, resources, store      # imported by everything, imports nothing
serving/     engine, recommend, resolve            # the read path: what should I watch
build/       pipeline, ingest, build_events, manifests
distribution/ bundle, releases, archives, profile_io, setup_status
cli.py       stays at the root — it is the entry point
```

This is a mechanical move plus import rewrites, and it should be done as one commit that
changes nothing else, so the diff stays reviewable.

### Two layering inversions worth fixing while the files are open

- `render/tables.py` imports `models.taste` — the terminal presentation layer reaches into
  model internals. It should take a rendered view object, not the model.
- `data/imdb.py` and `data/netflix.py` import `..pipeline` and `..store` — the lowest layer
  reaching upward into orchestration.

Both are small today. Both are exactly what an import contract exists to stop from growing.

### The frontend

`src/entertainer/web/ui/` is a Vite + React 19 app that builds into `web/static/`, which
FastAPI mounts. Keeping it inside the Python package is the right call for this project —
one `pip install`, one artefact, no second deploy — and the layout (`pages/`, `components/`,
`api.js`, `useAsync.js`) is conventional.

Two things are missing:

- **No tooling whatsoever.** No ESLint, no Prettier, no tests, no type checking. The Python
  half has `ruff` with a considered rule set; the JavaScript half has nothing at all.
- **Two files are outgrowing their box.** `pages/Home.jsx` (375 lines) and
  `components/Organism.jsx` (374 lines) carry fetching, state and presentation together.
  `ui.css` (351 lines) is a single global stylesheet.

### Tests

Fifty test files, flat in `tests/`, no `unit/` and `integration/` split and no mirroring of
the package structure. At fifty files this is still navigable; at a hundred it will not be.
Mirroring `tests/` to the new package layout is cheap to do at the same time as the move,
and expensive to do later.

### Conventions are not written down anywhere

There is no `CLAUDE.md` and no `AGENTS.md` in this repository. The README is 32 KB of
product and algorithm explanation — excellent, and not what a contributor needs to know
before touching the code. Nothing states the layering rules, the commit convention, where a
new feature goes, or how to run the checks. Every session re-derives it.

---

## 2. Dead code

`vulture` at 80% confidence finds **nothing**. At 60% it surfaces forty items, and thirty-two
of them are Typer command handlers — referenced by decorator, not by call, so they are false
positives by construction.

That leaves eight genuine candidates, each needing a one-line confirmation before removal:

| Symbol | Location |
| --- | --- |
| `CORE` | `bundle.py:47` |
| `REGION_TO_LANGUAGE` | `config.py:202` |
| `DEFAULT_EXPORT` | `profile_io.py:19` |
| `taste_summary()` | `models/discover.py:217` |
| `ucb_scores()` | `models/taste.py:226` |
| `Prediction` | `evaluation/personal.py:25` |
| `to_json_rows()` | `data/tmdb.py:345` |
| `fetch_imdb()`, `fetch_movielens()` | `data/download.py:99,111` |

Eight dead symbols across 22,000 lines is a good result, not a problem. Worth clearing
because it is an afternoon's work, not because it is urgent.

---

## 3. Dependencies

`deptry` over `pyproject.toml`, with the name-mapping noise (`sklearn` ↔ `scikit-learn`,
`dotenv` ↔ `python-dotenv`) discarded and each remaining finding confirmed by grep:

**Declared but never imported** — `pyarrow`, `tenacity`, `tqdm`, `platformdirs`,
`transformers` (in the `encode` extra). Five packages that every install pays for.

**Imported but never declared** — `threadpoolctl`, at `models/cf.py:162`. It works today
only because `implicit` happens to pull it in. The day `implicit` drops it, `ent data cf`
breaks on a machine that installed cleanly. This is the one real bug in this section.

---

## 4. Security

`pip-audit` across the installed set: **no known vulnerabilities**.

No hardcoded secrets anywhere in `src/`, `scripts/` or `tests/`. `.env` is ignored and has
never been committed. The `.gitignore` is unusually careful — it carries comments explaining
*why* `profile.jsonl` and an anchored `/data/` are excluded, both of which are real leak
paths for a public repository holding personal taste data.

The web auth design in `web/serve.py` and `web/app.py` is sound: a token is minted
automatically and cannot be opted out of once the server binds off loopback, the reasoning
is written down next to the code, and `docs_url` is disabled. Three hardening items remain,
all small:

1. **Token comparison is not constant time.** `web/app.py` compares with `!=`. Use
   `secrets.compare_digest`. Low practical risk on a LAN; it is a one-line fix.
2. **The token travels in the query string**, so it lands in browser history and any proxy
   log. The cookie hand-off is already implemented — redirect to the clean URL once the
   cookie is set.
3. **Fifteen f-string SQL sites** (`bundle.py`, `store.py`, `data/catalog.py`). All
   interpolate locally-derived paths and table names rather than user input, so none is
   exploitable today. Route them through parameters or a validated identifier allow-list so
   that stays true.

Confirmed *not* problems, recorded so they are not re-investigated: `subprocess` calls in
`releases.py` and `setup_status.py` pass argument lists with no shell and are safe; the
SHA-1 at `models/encoder.py:221` is a cache key, not a security primitive (mark it
`usedforsecurity=False` to silence the warning honestly); `slope != slope` in
`evaluation/prequential.py` is a deliberate NaN test and should become `math.isnan` for
readability, not correctness.

---

## 5. Tooling — buy this rather than build it

Nothing currently runs automatically. `ruff` is configured and passes, but only when someone
remembers to type it. There is no `.github/`, so there is no CI at all.

| Tool | Buys us | Cost |
| --- | --- | --- |
| **pre-commit** | Every tool below runs on commit, not on memory | Config only |
| **import-linter** | The layering rules become a failing test instead of a convention | Config only |
| **ty** (or **mypy**) | Type checking, which the project has none of | Incremental |
| **vulture** | Dead code, continuously | Config only |
| **deptry** | Dependency drift, continuously | Config only |
| **pip-audit** | CVEs, continuously | Config only |
| **ESLint + Prettier** | The frontend gets the same floor as the backend | Config only |
| **Vitest** | Frontend tests, which do not exist | Per test |
| **GitHub Actions** | All of it on every push | One workflow file |

`import-linter` is the important one. It is the answer to "enforce architecture boundaries":
the contracts are declared in config, and a violation fails CI rather than surviving until
someone notices in review.

Widening the `ruff` rule set is nearly free and was measured: adding `ARG`, `ERA`, `RET`,
`PTH`, `C4`, `TID`, `N`, `S`, `PL`, `TRY`, `A` and `DTZ` reports 667 findings, but the
distribution matters. 280 are `TID252` (relative imports — a deliberate house style, ignore
the rule), 112 are `PLC0415` (deliberate lazy imports that keep CLI startup fast, ignore),
and 59 are `TRY003`. Under a hundred findings across all rules are worth acting on.

---

## Today's work, in order

1. ~~**Write the conventions down**~~ *Done: `AGENTS.md` and `CLAUDE.md`.* — `CLAUDE.md` / `AGENTS.md` with the layering rules, where
   a new feature goes, the commit convention, and how to run the checks. Everything below
   depends on this existing, because it is what the tools will encode.
2. **Add `pre-commit` and a CI workflow** running `ruff`, `pytest`, `vulture`, `deptry` and
   `pip-audit`. Enforcement before cleanup, so the cleanup cannot silently regress.
3. **Fix the dependency list** — drop the five unused, declare `threadpoolctl`. Smallest
   real bug in the repository.
4. **Apply the three security fixes** — `compare_digest`, the query-string redirect, and the
   `usedforsecurity=False` annotation.
5. **Regroup the package root** into `core/`, `serving/`, `build/`, `distribution/`. One
   commit, imports only, no behaviour change, tests green before and after.
6. **Mirror `tests/` to the new layout** in the same commit.
7. **Add `import-linter` contracts** encoding the layering, and fix the two inversions
   (`render` → `models.taste`, `data` → `pipeline`/`store`) they will flag.
8. **Delete the eight dead symbols**, one confirmation each.
9. **Give the frontend a floor** — ESLint, Prettier, and a `lint` script wired into CI.
10. **Split `Home.jsx` and `Organism.jsx`**, extracting data fetching from presentation.
11. **Widen the `ruff` rule set** with the measured ignores, and clear what remains.
12. **Add a repository-layout section to the README**, or split it — 32 KB is past the point
    where a newcomer reads to the end.

Items 1 to 4 are a morning and carry the most value per minute. Items 5 to 8 are the actual
reorganisation. Items 9 to 12 are follow-on and can slip without blocking anything.

---

## Second pass — what the first audit missed

Added 25 September 2026. The first pass above measured the tree with tools: vulture,
deptry, pip-audit and ruff. This pass read the code for behaviour. Four reviewers covered
four areas in parallel: the store and build pipeline, the models and evaluation, the web
layer, and tests, docs and repo hygiene. Every finding below cites a file and line.
Each was checked against the code. Items marked *reproduced* were also run on a scratch
database, never on the real one.

This pass changes the headline. The first pass holds for layout and tooling. But the
repository has real correctness problems in three places that tools do not see:

- The rebuild path can attach verdicts to the wrong film.
- The web app returns errors when two requests overlap.
- A personal data file is committed to a public repository.

These are ranked ahead of every structural item. The regrouping in item 5 is worth doing,
but it does not protect a single verdict.

Severity: **critical** means personal data is exposed or verdicts are corrupted today.
**High** means a realistic path to wrong results or lost data. **Medium** means wrong
under specific conditions, or a trap for the next change. **Low** means tidiness.

### Corrections to the first pass

Fix these before item 7 (import-linter). Otherwise the contracts get written against the
wrong files.

- **Layering inversions are attributed to the wrong files** here and in `AGENTS.md`:
  - `data/imdb.py` and `data/netflix.py` import only `config` and `errors`. The real
    upward imports are `data/catalog.py:17-18` (`..pipeline`, `..store`) and
    `data/download.py:24` (`..archives`).
  - `render/tables.py` imports only `config`. The real one is a lazy import in
    `render/theme.py:26` (`to_display_scale`). `web/routers/slates.py:11` also uses that
    display helper. Move it to a neutral module, and the inversion is gone.
  - Missing from the list:
    - `evaluation/personal.py:14,16` imports `..engine` and `..manifests`.
    - `evaluation/simulate.py:294` imports `..engine`.
    - `offpolicy.estimate` takes an `Engine`.

    In each of these, the capability layer reaches up into the work layer.
  - `data/netflix.py:353` (`clear_previous`) writes to the store. Also,
    `data/catalog.py`, `imdb.py` and `tmdb.py` print to a rich `Console`. Both break "a
    data source parses and returns".
- **`render/theme.py:5` imports `typer`**, and `fail()` raises `typer.Exit`.
  `render/__init__` allows this. `AGENTS.md` forbids typer in library code and puts
  `render/` in the library layer. Pick one rule.
- **"Secrets and git hygiene: Clean. Nothing leaked" is false.** See S1.
- **The f-string SQL count is incomplete.** Beyond the 15 sites in `store`, `bundle` and
  `catalog`, there are more in:
  - `commands/build.py:467`
  - `resolve.py:133,149,189,200,256`
  - `engine.py:82`
  - `web/feed.py:107`

  None is exploitable. The `bundle.py:129` `ATTACH '{path}'` breaks on any data path
  that contains a quote.
- **The dead-code list is confirmed but incomplete.** All nine listed symbols are still
  present. Also dead:
  - `pipeline.STAGES` (tests only; `build_events.STAGES` is the live copy, with
    different labels)
  - `PATHS.interim` (created, never written)
  - `TasteModel.thompson_scores` (tests only)
  - `Engine.model(refit=False)` and `TasteModel.from_npz`: never called, so
    `taste.npz` is written and never read
  - the `snips(temperature=…)` parameter, and the logged-score tuple field (always 0.0)
  - `SimConfig.criterion` (overwritten by `run`)
  - the `except duckdb.IOException` branches at `releases.py:221` and
    `setup_status.py:270`. `_open` already translates that error, so these never run.
    See C3.
- **Organism.jsx does no fetching.** It is a pure canvas component, so item 10's split
  targets the wrong problem for it. Its real issues are W9 to W11.
- **The web auth hardening list is incomplete.** It misses loopback CSRF (S2), DNS
  rebinding (S3) and `/openapi.json` (S5). Items 1 and 2 are still accurate and unfixed.
- **Counts:** there are 48 test files, not fifty. The `slope != slope` checks sit at
  `prequential.py:268, 280, 306`, not in one place.
- **Item 1 is done.** `AGENTS.md` and `CLAUDE.md` exist. The section "Conventions are
  not written down anywhere" is now historical.

### S. Privacy and security

- **S1 — critical. `profile.jsonl` is committed to the public repository.** Commit
  `1f6e04c` ("chore: back up rating history", 21 Sep) added it. It holds 52 personal
  verdicts: IMDb ids, titles, scores and timestamps. It is on `origin/main`, and
  `gh repo view` reports `PUBLIC`. `.gitignore` lists the file, but ignore rules do not
  apply to a file that is already tracked. `git ls-files -ci --exclude-standard` shows
  it. This is the third hygiene violation, and it is not in `LEARNINGS.md`.
  - Fix: `git rm --cached profile.jsonl`.
  - Removing it from history means a history rewrite (`git filter-repo`) and a force
    push. That rewrite is irreversible and changes every later commit hash. It is the
    owner's decision.
  - Add the LEARNINGS line.
- **S2 — high. `/api/undo` can be forged from any website** (`web/routers/verdicts.py:59`).
  - The endpoint is a POST with no body. On loopback it needs no token, and nothing
    checks `Origin`.
  - `fetch('http://127.0.0.1:8756/api/undo', {method:'POST', mode:'no-cors'})` from
    any page deletes the newest web verdict. Called in a loop, it deletes all of them.
    *Reproduced*: `text/plain`, no body, 200.
  - Chrome's local-network prompt may block this. Safari and Firefox do not.
  - `/api/recommendations/slate` can be forged the same way. The forgery fills the
    off-policy log with fake slates.
  - Fix: middleware that rejects any non-GET request without `application/json` or
    with a foreign `Origin`. Undo takes an explicit `event_id` (see W7).
- **S3 — medium. DNS rebinding reads the whole taste profile** (`web/app.py:53-101`,
  `web/studio.py:21`).
  - Neither app validates `Host`. *Reproduced*: `Host: evil.example:8756` returns 200.
  - A rebinding page can read `/api/rated` and `/api/progress` and post verdicts.
  - Fix: `TrustedHostMiddleware` allowing loopback names plus
    `binding.display_host`, in both apps.
- **S4 — low. `AddRequest` accepts any `kind` and `verdict`** (`web/schemas.py:23-26`).
  - A bad verdict raises inside `engine.record` after the title is inserted and
    `content.npy`/`fused.npz` are rewritten, so the add is left half-done.
  - `kind: "foo"` is written into `titles.kind`.
  - Fix: `Literal` types, validated before any write.
- **S5 — low. `/openapi.json` is still served** (`web/app.py:53`). `docs_url` and
  `redoc_url` are off, but the schema is not. Set `openapi_url=None`.
- **S6 — low. The `langs` query weights are unvalidated** (`routers/catalogue.py:97-106`
  → `live.py:118-123`, `feed.py:69`).
  - `hi:inf` yields NaN and a 500.
  - Thousands of codes in live mode fan out an unbounded `asyncio.gather` against the
    user's TMDB key.
  - Fix: clamp the weights and cap the number of codes.
- **S7 — low. The TMDB search error returns raw exception text to the browser**
  (`routers/catalogue.py:189-212`), against the rule in `failures.py`. The same block
  duplicates `_tmdb_matches` (216-249).

### D. Data integrity — the verdict log

- **D1 — critical. A rebuild reattaches orphaned verdicts to other films**
  (`data/catalog.py:161-190`, *reproduced*).
  - When a rebuild drops a rated title, its events keep the old positional `item_id`.
    A new title now has that id. The log line "verdicts are preserved but inactive" is
    false: they are active, on the wrong film.
  - `imdb.build` has no exemption for rated titles. So a rebuild drops these, and their
    verdicts move:
    - every hand-added title (`ent add`, Netflix import)
    - any rated title under the vote floor
    - any rated title of an excluded type

    This also breaks the AGENTS rule "never prune a title the user has rated".
  - Fix: exempt titles that have events from the IMDb filter, and carry them across
    as `_carry` does. Refuse the rebuild if any event would still be orphaned, as
    `bundle._replace_titles` does.
- **D2 — high. The rebuild is not transactional** (`data/catalog.py:152-184`).
  - `DELETE FROM titles`, the INSERT and the remap run in autocommit, and `_old_ids`
    is a TEMP table.
  - A crash or Ctrl-C between the INSERT and the UPDATE leaves every verdict on
    renumbered ids. The only old-to-new map dies with the connection.
  - Fix: BEGIN/COMMIT around the block, as `bundle._replace_titles` already does.
- **D3 — high. The rebuild remaps 2 of the 5 places that hold item ids**
  (`data/catalog.py:180`).
  - It remaps `events` and `impressions`.
  - It skips `validation_cases.item_id`, the id lists in `validation_batches` and
    `meta.last_slate`.
  - `bundle._replace_titles:341-377` remaps all five, so the two routines have drifted.
  - After a rebuild:
    - `ent loved 3` resolves to a different film.
    - Sealed predictions reveal against the wrong title.
    - The UNIQUE on `validation_cases.item_id` can collide.
  - Fix: one shared remap helper, used by both routines. D1 to D3 are one piece of work.
- **D4 — high. TMDB failures are recorded as "enriched, nothing found"**
  (`data/tmdb.py:129-150`, `data/catalog.py:480,505`).
  - `_get` returns `None` for 401, for exhausted retries and for other 4xx, exactly as
    for a real 404.
  - `apply_enrichment` then stamps `enriched_at` on every row. `pending_enrichment`
    selects `enriched_at IS NULL`, so those rows are never retried.
  - So a revoked key, an outage or a network drop during the multi-hour pass marks
    thousands of titles done with no data. The language prune then judges them on the
    default floor. Keywords have the same flaw (`_keywords_only` → `apply_keywords`).
  - Fix: distinguish "not found" from "did not answer", and stamp only rows TMDB
    actually answered.
- **D5 — high. The profile export silently drops verdicts** (`profile_io.py:76-82`).
  - An INNER `JOIN titles` omits every event whose title is gone. D1's orphans are
    exactly such events. The export exists to survive rebuilds, and it loses precisely
    what a rebuild orphans.
  - Hand-added titles with a NULL `imdb_id` export with null ids and fail on import.
  - The returned count hides the loss.
  - Fix: LEFT JOIN, and report or refuse unmatched rows.
- **D6 — medium. The profile export is not atomic** (`profile_io.py:82`). It writes over
  `profile.jsonl` in place, so a crash truncates the only durable backup. Write to a
  temp file, then `os.replace`.
- **D7 — medium. `ent netflix --replace` deletes before it re-imports**
  (`data/netflix.py:353-362`, `commands/transfer.py:209-230`).
  - Every `source='netflix'` event is hard-deleted first. Each failed `add_title` is
    then skipped.
  - With TMDB down, all Netflix verdicts are gone and none come back.
  - Fix: one transaction, committed only after resolution succeeds.
- **D8 — medium. A bundle restore leaves stale positional files**
  (`bundle.py:282-294`).
  - Only the files present in the bundle are replaced. `bundle export` defaults to
    `encodings=False`, and it omits a missing `population_prior.npz` or `fused.npz`
    silently.
  - The old `content_ids.npy`/`content.npy` then point at the previous catalogue's
    ids. `ingest.add_title` appends to them, and the next `ent data fuse` builds on
    wrong vectors.
  - Fix: delete or move aside every space and encode file the bundle does not carry.
- **D9 — medium. The array appends are unlocked and non-atomic**
  (`ingest.py:46-58`, `web/routers/additions.py:50-63`, `models/encoder.py:373-374`,
  `fusion.save`).
  - Two concurrent adds each load the old arrays, and the last write wins. One title
    exists in the catalogue but can never be recommended.
  - `np.save` truncates in place. A concurrent `fusion.load()` can read a half-written
    file. A crash between the ids file and the matrix file misaligns them.
  - Fix: a process lock plus temp-file-then-replace writes.
- **D10 — medium. Verdict-log INSERTs are positional** (`store.py:209,241,249,283,300`;
  `evaluation/personal.py:126`). This breaks the AGENTS rule on the most important
  table.
- **D11 — low. `NOT IN (SELECT item_id FROM events)`** (`data/catalog.py:366`,
  `store.py:334`).
  - A single NULL `item_id` makes the whole subquery NULL, so the prune deletes nothing
    and `negatives()` returns nothing.
  - The prune also ignores `validation_cases` and `impressions`.
  - Fix: `NOT EXISTS`.

### C. Concurrency and processes

- **C1 — high. The web app returns 500 when requests overlap**
  (`store.py:158-161,196`, every router; *reproduced*).
  - Handlers are sync, so FastAPI runs them in a threadpool. They mix `read_only=True`
    and read-write connections to one file in one process.
  - DuckDB refuses that with `ConnectionException: Can't open a connection to same
    database file with a different configuration`. That error is not an
    `IOException`, so it is never translated into `CatalogueBusy`.
  - Measured: 40 rates against 40 progress reads gave 45 responses of 500. Every call
    succeeds alone.
  - In the app: rating quickly in Focus mode shows "Something went wrong", and the
    verdict is lost. `/api/rate` alone opens both modes (`verdicts.py:21,37,51`).
  - Fix: one connection mode in the web process, or one app-scoped connection with a
    `.cursor()` per request. Add a concurrency test.
- **C2 — high. The web app never notices a rebuild** (`engine.py:69-121`,
  `web/app.py:73`, `web/context.py:49`).
  - One `Engine` lives for the app's lifetime. `_meta`, `_fs` and `_prior` never
    invalidate. `additions.py:62` pokes the private fields and leaves `_prior` alone.
  - After `ent setup`, `ent pull` or a Studio build, the running app scores on old ids
    and fetches rows by new ids. It shows the wrong titles, and a verdict on a
    displayed id lands on another film.
  - Fix: stamp the build id and `fused.npz` mtime in `Engine`. Check the stamp on
    access. Add a public `invalidate()`.
- **C3 — high. A build can die hours in on a lock** (`store.py:148-169`).
  - `_open` does not retry. `ent setup` opens about ten connections
    (`commands/build.py:181,225,380,416,467,554,589`).
  - Studio polls `/api/setup` → `setup_status._this_machine:268`, which opens the
    database. Any overlap kills the build with `CatalogueBusy`.
  - The Studio panel itself then returns 500, because `setup_status.py:270` catches
    the untranslated `IOException`.
  - Fix: bounded retry with backoff on write opens, and catch `CatalogueBusy` in
    `setup_status` and `releases`.
- **C4 — low. The first start can race** (`store.py:180-185`).
  - App and Studio started together on a fresh machine both run bootstrap. One fails
    with `CatalogueBusy`.
  - The bootstrap connection also leaks if SCHEMA raises.
- **C5 — low. Failures between build stages go unrecorded**
  (`build_events.py:156-166`).
  - `reporter.stage()` records a stage's failure. Anything raised between stages
    leaves `state.json` at "running", with no error text.
  - Studio then shows only "interrupted".
  - Fix: wrap the whole `setup` body.

### N. Network and external calls

- **N1 — medium. Dump downloads have no timeout and no retry**
  (`data/download.py:74`). `timeout=None`, so a stalled CDN hangs `ent setup` stage 1
  forever. Resume logic exists; retry through it.
- **N2 — medium. `gh` calls have no timeout** (`releases.py:55,70`). A stalled
  `ent pull` or `publish` hangs forever. A hung pull also leaves the app stopped.
- **N3 — low. MovieLens is "done" once `ratings.csv` exists**
  (`data/download.py:119-124`). A crash mid-extract leaves `links.csv` missing forever.
  The CF identity join is then silently skipped.
- **N4 — low. The published build id is scraped from the release body**
  (`setup_status.py:28,244`). `--notes` replaces the body, so status then reads "not
  published yet" forever. Read the attached `bundle.json` instead.
- **N5 — low. The build meta is recorded after the upload** (`releases.py:121-124`). A
  lock error at that point leaves a real release that the builder reports as
  unpublished.

### M. Model and evaluation correctness

- **M1 — high. The off-policy estimate is meaningless**
  (`evaluation/prequential.py:344-346`, `offpolicy.py:110-118`,
  `recommend.py:260-263`).
  - The model is fit on every verdict, including the rewards being evaluated. That
    is in-sample, so "better" is biased towards yes.
  - The new policy uses softmax at T = 1.0 over scores in 0..1, which is nearly
    uniform. The logging policy used T = std(scores) per slate.
  - New probabilities are normalised over the pooled log, not per slate.
  - Fix: fit only on verdicts before each impression. Normalise per slate with the
    logger's temperature rule, and log that temperature with the impression.
- **M2 — medium. The off-policy join over-counts** (`offpolicy.py:86-99`).
  - Every impression matches every later rating of the item.
  - A title shown in 5 slates and rated once counts 5 times.
  - `claimed` works per slate, not per (slate, item), which contradicts the comment at
    64-65.
- **M3 — high. Title resolution picks the wrong film for titles that end in a year**
  (`resolve.py:23,40-44,120-124,187-224,242`).
  - `split_year` treats any trailing 19xx/20xx as a year.
  - "Blade Runner 2049" becomes "blade runner" with a year of about 2049. Tiers 1 and 2
    find nothing. Tier 3 then scores Blade Runner (1982) at 0.96, accepts it as
    confident, and records the verdict silently on the wrong film. The module
    docstring names this exact failure as the one to avoid.
  - "Wonder Woman 1984" and "Class of 1984" hit it too.
  - Fix: when the year-filtered tiers find nothing, retry on the whole string.
- **M4 — medium. `ent audit` measures a model the engine does not serve**
  (`evaluation/prequential.py:169`; history built twice, at
  `commands/diagnose.py:100-108` and `web/routers/insight.py:92-101`).
  - The audit fits with no negatives, no prior, no decay and no skip weights. That is
    the configuration `engine.py:47-53` records as degenerate.
  - The audit keeps the first verdict per item. The engine uses the latest.
  - Fix: one pure `fit_from_labels(...)`, shared by engine, prequential, simulate and
    personal.
- **M5 — medium. Negative sampling exists in three drifted copies**
  (`engine.py:269-295`, `evaluation/simulate.py:284-305`,
  `evaluation/personal.py:37-50`).
  - Only simulate honours `ENTERTAINER_NEGATIVE_SAMPLES`. Personal hardcodes 1000.
  - Simulate ignores decay and skip weights. Personal treats skips as full verdicts.
  - `ent eval` therefore benchmarks a model that is not the serving model.
- **M6 — medium. The simulator scores known ratings as misses**
  (`evaluation/simulate.py:412-421`).
  - Unasked items from the `known` split stay among the candidates, but relevance comes
    only from `held`.
  - Every arm is deflated. The arms that find taste best are penalised most.
- **M7 — medium. The eval holdout is not tied to the prior**
  (`evaluation/integrity.py:71-92`, `commands/build.py:524-535`).
  - Fit the prior with holdout 0, then run `ent data cf --holdout 2000`. The stale
    prior passes the integrity check, and `ent eval` reports a leaked number.
  - Fix: store the holdout hash in the prior's npz.
- **M8 — medium. One "haven't seen it" answer blocks a sealed batch forever**
  (`evaluation/personal.py:215` vs 147-149). The completeness check ignores unseen
  cases, so `DECISION_CASES` may never be reached.
- **M9 — medium. `--strategy ucb` discards mood and novelty** (`recommend.py:243-252`).
  `scores[top] = adjusted` overwrites the adjusted scores.
- **M10 — medium. The CLI builds impressions by hand** (`commands/browse.py:179-190`).
  - `produce_slate`'s docstring says the CLI uses it. It does not.
  - The CLI's exclusion set differs from the web's (no watchlist).
  - `strategy` is not validated: `--strategy ubc` runs Thompson but logs `ubc-n169`,
    which corrupts off-policy grouping.
- **M11 — medium. Each RFF slate peaks near 2 GB** (`recommend.py:195-197`,
  `models/taste.py:127,134-140`).
  - `W` is float64, so `X @ W` upcasts a 270k-row matrix. Fancy indexing also copies
    about 400 MB.
  - Concurrent web requests multiply the peak.
  - Fix: float32 `W`/`b`, and chunked scoring.
- **M12 — low. Smaller model issues:**
  - `engine.py:174,189`: skips and dismissals count as verdicts in `n_real` and in the
    "fewer than 3" gate.
  - `recommend.py:137`: MMR mixes raw reward-scale scores with cosines under a fixed
    λ = 0.72, so diversity drifts with temperature.
  - `resolve.py:135-137`: tier-1 SQL normalisation does not match `normalise()`.
    "spider man no way home" misses tier 1.
  - `evaluation/metrics.py:64,92-94`: the whole popularity dict is sorted and summed
    per user per arm.
  - `simulate.py:345,370`: `_ACTIVE_PRIOR` is a module global that is never reset.
  - `why`, `taste` and `onboard` rewrite `taste.npz` on read-only commands. Nothing
    reads the file.
  - `personal.py:82-87` re-implements `to_display_scale`.
  - `prequential.py:92`: `spearmanr` on constant input raises `ConstantInputWarning`
    in production.
- **M13 — low. The model-not-ready message names a missing command**
  (`engine.py:95-96`). "run `ent build`" names a command that does not exist.
  `ent setup` is the right one.

### W. Frontend

- **W1 — medium. `useAsync` has a stale-response race** (`ui/src/useAsync.js:12-28`).
  - The `alive` ref guards against unmount only. No `AbortController` exists anywhere.
  - On Rate, a slower older feed overwrites a newer one. AGENTS tells contributors to
    use this hook everywhere, so the convention spreads the bug.
  - Fix: a per-run id or an abort signal passed through `request()`.
- **W2 — medium. Rate's keyboard shortcuts misfire** (`pages/Rate.jsx:121-134`).
  - Holding `l` rates title after title at key-repeat speed.
  - Typing in the `<select>` also rates a title.
  - Fix: ignore `event.repeat` and form targets, and respect `busy`.
- **W3 — medium. The blind test cannot be reached from either front end**
  (`ui/src/api.js:58-61`, `pages/Evidence.jsx:210`).
  - Evidence tells the user to seal and reveal titles. No page and no CLI command can
    do it.
  - `api.js` also defines `rated`, `similar`, `predict` and `checkTitles`, and nothing
    calls them.
- **W4 — medium. Muted text fails contrast** (`ui/src/theme.css:9,38,74`).
  - `--fg-3` on the background is about 2.3:1, below the 4.5:1 WCAG AA minimum.
  - It is used for 9px labels (22 uses) and the nav.
- **W5 — low. Screen-reader gaps** (`TitleCard.jsx:94-104`, `Rate.jsx:155-161`,
  `Recommendations.jsx:86-95`, `Home.jsx:261`):
  - 60 identical "Loved" buttons with no title in their names
  - no `aria-pressed` on toggles
  - no `aria-activedescendant` on the listbox
- **W6 — low. Errors are swallowed.**
  - Rate undo, Library actions and Recommendations drop errors silently.
  - `validation.error` is never rendered (`Rate.jsx:100`, `Library.jsx:63`,
    `Recommendations.jsx:34,52`, `Evidence.jsx:135`).
- **W7 — low. Undo removes the newest web verdict, not the one the toast names**
  (`routers/verdicts.py:59-77`). This is wrong with a request in flight or a second
  tab. Return an `event_id` from `/api/rate` and undo that event.
- **W8 — low. Evidence calls hooks after an early `return`** (`pages/Evidence.jsx:26`
  vs 48-56).
  - It will throw "Rendered more hooks" the first time `curve` fills on a live mount.
  - The effect also has no deps array.
- **W9 — low. Organism ignores reduced motion for its main loop**
  (`components/Organism.jsx:83,161-166`). Wobble, twinkle and the rAF loop run forever.
  TasteField handles reduced motion correctly.
- **W10 — low. Organism re-seeds on every prop change** (`Organism.jsx:371`).
  - The effect depends on six props, against its own comment at 67-69.
  - Recommendations flips two of them per load, so the creature jumps.
- **W11 — low. The canvas animations are tied to frame rate**
  (`Organism.jsx:153,161`, `TasteField.jsx:113`).
  - Time advances a fixed step per frame, so animation runs twice as fast at 120 Hz.
  - About 2,800 `arc` calls per frame, even when idle.
  - Fix: use the rAF timestamp delta, and pre-render static layers.
- **W12 — low. Vite does not empty its output directory**
  (`web/ui/vite.config.js`, `emptyOutDir: false`). Old hashed bundles pile up in
  `web/static/assets/`; history already holds 6 generations. Prune unreferenced assets
  after a build.

Checked and fine: the committed `web/static` bundle is byte-identical to a fresh build of
`ui/src`. `npm audit` reports 0 vulnerabilities, a lockfile is present, and all runtime
deps are used. Static serving is safe from traversal. No `async def` handler blocks. The
500 handler does not leak tracebacks.

### A. Front-end asymmetries

`AGENTS.md` calls a feature that only one front end can reach a design mistake. Today:

- **Web only:**
  - library save, remove and watched
  - undo
  - check-titles
  - judge on a TMDB-only title
  - predict
  - similar-by-id
- **CLI only:**
  - `dismiss`, `forget`, `history`, `bulk`
  - `onboard`, `why`, `stats`, `eval`
  - profile `export`/`import`, `netflix`
  - `bundle`/`release`/`data`
- **Neither:** validation seal, reveal and cases (W3).
- **Differs:** CLI `seen` logs the event but does not end the watchlist entry. Web
  "watched" does both.

Not every item needs a twin; the maintenance verbs are reasonably CLI-only. The list is
here so the call is made deliberately.

### T. Tests

Baseline: `534 passed, 6 warnings in 65.79s`, stable in shuffled order. Line coverage
is 81% with branches.

- **T1 — high. The packaging guard misses tracked-but-ignored files**
  (`tests/test_packaging.py::test_secrets_and_personal_data_stay_ignored`).
  `git check-ignore --no-index` checks patterns only, which is why S1 passed. Add an
  assertion that `git ls-files -ci --exclude-standard` is empty.
- **T2 — high. The documented install omits the `web` extra.**
  - `README.md:385` and `AGENTS.md:249` say `.[encode,dev]`, so fastapi is missing.
  - On such an install:
    - `ent rate`, `ent studio` and `./scripts/entertainer start` fail.
    - `tests/test_build_studio.py:5` breaks collection.
    - 81 web tests silently skip.
  - `scripts/entertainer:13` says `.[web]`, so three instructions disagree.
  - Fix: document `.[encode,web,dev]`, and guard test_build_studio with
    `importorskip`.
- **T3 — medium. Nothing isolates tests from the real data directory**
  (`tests/conftest.py`).
  - `ENTERTAINER_DATA_DIR` defaults to `data/`, where the real 733 MB database lives.
  - Only 12 of 48 files opt in to a tmp dir. This has already overwritten `taste.npz`
    once.
  - Fix: an autouse session fixture.
- **T4 — medium. The commands people actually run are barely tested:**

  | Module | Coverage |
  | --- | --- |
  | `commands/serve.py` | 20% |
  | `commands/transfer.py` | 26% |
  | `web/routers/additions.py` | 30% |
  | `web/live.py` | 32% |
  | `commands/build.py` | 34% (the `ent setup` orchestration) |
  | `commands/release.py` | 40% |
  | `commands/diagnose.py` | 55% |

  Fix: CliRunner smoke tests with the work functions monkeypatched, and a TestClient
  test for `/api/add`.
- **T5 — medium. There is no concurrency test.** C1 would have been caught by one.
  D1 to D3 need a rebuild test that drops a rated title and checks that every id-bearing
  table survives.
- **T6 — low. Timing-based tests:**
  - `test_build_overlap.py:31` (`sleep` + `assert peak > 1`)
  - `test_web_failures.py:128-151` (10 s deadline, leaks the `Popen` pipe:
    `ResourceWarning`)
  - `test_tmdb_enrich.py:117`

  All pass today. Barriers and Events would make them deterministic.
- **T7 — low. The web app fixture is rebuilt for every test**, about 0.4 s each. The
  fixture could be module-scoped.

### R. Repository, docs and tooling

- **R1 — medium. `ruff format` has never been applied.**
  - `ruff format --check` reports 80 files to reformat and 44 clean, about 2,660
    lines.
  - Either do one mechanical format commit before item 2 adds enforcement, or state in
    `AGENTS.md` that format is not enforced.
- **R2 — medium. `uv.lock` is neither committed nor ignored.**
  - `pyproject.toml` pins torch to a CUDA index through `[tool.uv.sources]`, but the
    docs use `uv pip install`, which ignores the lock.
  - Builder and puller machines drift, and a stray `git add -A` commits the file.
  - Fix: commit it and document `uv sync --extra encode --extra web --extra dev`, or
    ignore it.
- **R3 — medium. `docs/BACKLOG.md` §9 is stale.**
  - It says release publish and pull have never been used.
  - `sarathkumar365/entertainer-builds` exists, and release `build-20260925-1806` was
    published today.
  - §2 says "web half is done", then "nothing does this end to end today".
- **R4 — low. Duplicated constants:**
  - `STAGES` exists in `build_events.py:23` and `pipeline.py:99`.
  - `REPO_ROOT` is redefined in `setup_status.py:27`.
  - `bundle.py:38` imports the private `manifests._digest`, and
    `releases._download:160-163` re-implements that sha256 loop.
  - 192 is repeated in `BuildSettings`, `config` and the typer defaults.
- **R5 — low. `.env.example` is missing keys the code reads:**
  - `ENTERTAINER_ROLE`: documented nowhere
  - `ENTERTAINER_MEMORY_FRACTION`, `ENTERTAINER_DEVICE`, `ENTERTAINER_CF_GPU`: README
    only
  - `ENTERTAINER_NEGATIVE_SAMPLES`: RESULTS only
  - `ENTERTAINER_NO_DOTENV`

  List them, commented out.
- **R6 — low. `.claude/` is only partly ignored.**
  - `settings.local.json` is ignored only by the author's global git config.
  - `.claude/.reviewed` and `.claude/worktrees/` are not ignored.
  - Add all three to `.gitignore`.
- **R7 — low. Root `reports/` is tracked**, yet `PATHS.reports` is `data/reports` and
  the AGENTS layout omits it.
  - `rate.log` is a captured banner.
  - The bench logs are cited by BACKLOG, so move them to `docs/evidence/` and ignore
    `/reports/`.
- **R8 — low. `scripts/finish_build.sh` predates the resumable `ent setup`.**
  - No doc mentions it.
  - It uses cwd-relative `data/raw/...` and ignores `ENTERTAINER_DATA_DIR`. Run from
    outside the repo, it waits about 90 minutes for a file it will never see.
  - `scripts/entertainer:8` also hard-codes `$ROOT/data/runtime`.
  - Delete the script, or fix the paths.
- **R9 — low. `ruff check ... scripts` lints nothing.** `scripts/` holds only bash.
  Add shellcheck to the pre-commit in item 2.
- **R10 — low. Oversized functions.** Line counts, excluding docstrings:

  | Function | Lines |
  | --- | --- |
  | `catalog.build_base` | 121 |
  | `commands/build.py:setup` | 118 |
  | `resolve.search` | 115 |
  | `transfer.netflix_import` | 103 |
  | `recommend.recommend` | 96 (14 parameters) |
  | `browse.recs` | 91 |
  | `bundle._replace_titles` | 90 |
  | `diagnose.evaluate` | 81 |
  | `simulate.run` | 79 |

  `build_base` and `_replace_titles` hold the drifted remap logic behind D1 to D3.
- **R11 — low. Duplicated scoring maths:**
  - Predictive variance is computed three ways, with different jitter:
    `taste.py:194`, `recommend.py:101-111` (re-factors Cholesky per call, ignoring the
    cached `_chol`) and `elicit.py:252,298`.
  - `recommend.py:212-213` inlines `thompson_scores`.
- **R12 — low. Magic numbers that belong in config:**
  - `reference_year=2026` (`features.py:99,160`)
  - `i >= 150` vs `RFF_MIN_OBS = 120` (two thresholds for one decision)
  - MMR 0.72, shortlist 400
  - resolve thresholds 0.95/0.08/0.75/0.20
  - elicit caps 1200/400

### Revised order

The first pass's list stands for its own items. The new work slots in ahead of it:

1. **S1**: untrack `profile.jsonl`. The owner decides on the history rewrite. Add T1 in
   the same change.
2. **S2, S3, S5** with the existing security item 4: one web-hardening change.
3. **C1**, with its concurrency test (T5). This is the most user-visible bug.
4. **D1 to D3** together: one shared remap helper, one transaction, and a rebuild test
   that drops a rated title. Then **D5 and D6**, so the backup cannot lose what a
   rebuild orphans.
5. **D4**: stop stamping TMDB failures as enriched. Titles already stamped wrongly need
   a one-off reset of `enriched_at` for rows with no `tmdb_id`.
6. **C2 and C3**: engine invalidation, and lock retry.
7. **M3** (resolve) and **M13** (wrong command in message). Both are small and user
   facing.
8. **T2 and T3**: correct the install docs, and isolate tests. Then the first pass's
   items 2 and 3, with **R1 and R2** decided first so enforcement starts from a clean
   tree.
9. **M1, M2, M4, M5**: the evaluation needs to measure the model that actually serves
   before any of its numbers are trusted for a decision.
10. Everything else, by severity, alongside the first pass's items 5 to 12. D10 and the
    corrected inversions belong with item 7. R4, R10 and R11 belong with item 5.
