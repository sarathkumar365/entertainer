# Code health — audit and cleanup plan

Written 25 September 2026, after a full read of the repository against the question the
project has never asked itself: *is this laid out the way a stranger would expect?*

The project was scaffolded and built fast, feature by feature, and nothing has been
reorganised since. That shows up in exactly one place — the package root — and almost
nowhere else. This document records what the audit found, what is already fine and should
be left alone, and the ordered work for turning the findings into enforced rules rather
than good intentions.

Every claim below was produced by running a tool against the tree, not by impression.

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

1. **Write the conventions down** — `CLAUDE.md` / `AGENTS.md` with the layering rules, where
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
