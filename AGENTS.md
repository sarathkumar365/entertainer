# Contributing to entertainer

The operating contract for this repository: where code goes, which direction
dependencies run, what the conventions are, and what must never be committed.

The README explains what the product is and how the engine works. This file explains how
the repository is built. Read this before touching code; read the README before touching
the model.

---

## What this is

A personal, self-improving recommendation engine for movies and TV across multilingual
catalogues. It runs entirely on one machine. Taste data never leaves it.

Two halves: **the engine**, which learns from titles and verdicts alone (built), and **the
agent**, which acts on the engine's answers to get titles into a Jellyfin library (not built
— see the README, "Part two").

**This repository is public. The data it processes is private.** That tension is the source
of the hardest hygiene rules below, and it has already been violated twice — both times
recorded in `docs/LEARNINGS.md`.

---

## Layout

```
src/entertainer/
  cli.py          entry point named in pyproject; stays a module, not a package, so
                  `git log --follow` keeps working on the file with the most history
  config.py       paths, tunables, environment overrides
  errors.py       domain errors whose message is written for a person
  resources.py    machine-capability decisions (memory share, DuckDB config)
  store.py        DuckDB persistence: catalogue plus the interaction log
  engine.py       the read path's facade
  recommend.py    slate production
  resolve.py      turning a typed title into an item
  pipeline.py     build orchestration
  ingest.py       build entry into the store
  build_events.py build progress as events
  manifests.py    what a build produced, and from what
  bundle.py       catalogue bundles for moving between machines
  releases.py     publish and pull via the GitHub CLI
  archives.py     raw dump handling
  profile_io.py   verdict export and import
  setup_status.py whether this machine can build, publish and pull

  data/           ingest and enrichment — imdb, netflix, tmdb, catalog, download
  models/         the learning — cf, encoder, features, fusion, taste, discover,
                  population, itemcard
  evaluation/     the measuring — simulate, prequential, personal, offpolicy, metrics,
                  integrity
  coldstart/      elicitation — elicit, session
  render/         terminal presentation — tables, theme
  commands/       the CLI verbs, grouped by purpose
  web/            FastAPI app, routers/, and the React app in ui/

tests/            pytest, flat, with golden CLI output in tests/golden/
docs/             see "Which document" below
scripts/          the local control surface (`./scripts/entertainer`)
data/             gitignored. Raw dumps, artefacts, runtime state, reports
```

The eighteen root modules are grouped above by layer, but they are physically flat. That is
a known problem with a plan attached: `docs/CODE_HEALTH.md` item 5 regroups them into
`core/`, `serving/`, `build/` and `distribution/`. Until that lands, add new root modules
sparingly, and prefer a subpackage.

---

## Dependency direction

Four layers. Imports run downward only.

```
commands/  web/          entry points — may import anything below
    ↓
engine  recommend  resolve  pipeline  ingest     the work
    ↓
data/  models/  coldstart/  evaluation/  render/  the capabilities
    ↓
config  errors  resources  store                  infrastructure — imports nothing local
```

Rules that follow from this, all of them currently true or nearly so:

- **`models/`, `coldstart/` and `data/` import almost nothing but `config` and `errors`.**
  Two known violations exist — `data/imdb.py` and `data/netflix.py` reach up into
  `pipeline` and `store`. Do not add a third.
- **`render/` renders. It does not reach into model internals.** `render/tables.py`
  currently imports `models.taste`; it should take a view object. Do not extend the pattern.
- **Library code must not import `typer`.** A module that raises `typer.Exit` can only be
  called from a command, which puts business rules out of reach of the web app and of
  tests. Raise a domain error from `errors.py` instead. `cli.main` catches
  `EntertainerError` at the boundary and prints one red line rather than a traceback;
  `commands/_shared.py` holds the helpers that do talk to a person, and those may raise
  `typer.Exit`.
- **`web/` and `commands/` are two front ends over the same functions.** A feature that
  only one of them can reach is a design mistake, not a scope decision.

---

## Where new code goes

| You are adding | It goes in |
| --- | --- |
| A new CLI verb | `commands/`, in the group it belongs to; register it in `commands/__init__.py` |
| A new HTTP endpoint | `web/routers/`, in the router it belongs to |
| A new data source | `data/`, as a module that parses and returns — it does not write |
| A new model or scorer | `models/` |
| A new measurement | `evaluation/` |
| A new build stage | `pipeline.py`, emitting through `build_events.py` |
| A shared helper used by two packages | the lowest layer that both can see, never a new root module |
| A new screen | `web/ui/src/pages/`, with its fetch through `api.js` and `useAsync.js` |

If a helper is needed by exactly one caller, it lives next to that caller. Utility modules
are earned by a second caller, not anticipated.

---

## Style

- **Python 3.11+, line length 100, ruff with `E,F,I,UP,B,SIM`.** `ruff check` must pass
  before a commit. It currently does, from a clean tree.
- **`from __future__ import annotations` at the top of every module.**
- **Relative imports within the package** (`from ..config import PATHS`). This is a house
  style, deliberately at odds with ruff's `TID252`, which is why that rule is not enabled.
- **Lazy imports are deliberate where they appear.** `torch` and `sentence-transformers`
  take seconds to import; the CLI must start faster than that. Keep heavy imports inside
  the function that needs them, and leave the ones that are already there alone.
- **Module docstrings say why, not what.** Every module in this repository opens with the
  reasoning behind its existence — which decision it encodes, what was tried and rejected.
  Match that. A docstring that restates the module name is worse than none.
- **Comments explain decisions, not mechanics.** The comment density here is high and the
  content is unusual: most comments record a trap. Keep them that way.
- **Errors carry a next step.** Every class in `errors.py` exists because a state of the
  system was being reported as a defect. A new error message should tell the reader what to
  do, in a sentence, without a stack trace.

### Frontend

React 19 + Vite, in `src/entertainer/web/ui`, building into `web/static/`. The built bundle
is committed so that running the app needs no `node`; editing it does.

```bash
./scripts/entertainer ui     # installs deps on first run, then builds
```

There is currently no linter, formatter or test runner on this half of the codebase —
`docs/CODE_HEALTH.md` item 9. Until that lands, match the surrounding files: fetching
through `api.js`, async state through `useAsync.js`, one page per screen, shared pieces in
`components/`.

---

## Tests

```bash
.venv/bin/pytest               # the whole suite
.venv/bin/pytest tests/test_web.py -x
.venv/bin/ruff check src tests scripts
```

- Tests live flat in `tests/`, named `test_<subject>.py`.
- CLI output is checked against golden files in `tests/golden/`. Changing rendered output
  means updating them deliberately, not regenerating them blindly.
- `tests/test_packaging.py` asserts the gitignore invariants — that no source or test
  file is excluded, and that every secret path is. `tests/test_data_directory_containment.py`
  asserts that every module binds its paths through `config` rather than at import time.
  Both exist because the thing they check failed in production. Do not weaken them.
- A bug fix carries the test that would have caught it.

---

## Committing

Conventional Commits, with a subject that states the **effect**, in lower case, as a person
would say it:

```
fix(eval): replay only the users held out of training
feat: ask "Will I like it?" about any title from Home
refactor: split the CLI into command groups
docs: restate what the platform is for
```

The body explains **why**, and what was rejected. Bodies here run several paragraphs when
the change earned it — see `34332c8` for the shape. End with the `Co-Authored-By` trailer.

Work happens on a branch per change, merged with a merge commit whose subject names the
branch and its effect:

```
Merge feat/will-i-like-it: ask about any title from Home
```

Never commit directly to `main` for anything larger than a typo.

---

## Which document

| File | Holds |
| --- | --- |
| `README.md` | What the product is, how the engine works, how to install and use it |
| `docs/HOW_IT_WORKS.md` | The plain-English walkthrough for a non-technical reader |
| `docs/BACKLOG.md` | Work, ordered by the order it should be done in |
| `docs/LEARNINGS.md` | Failures that actually happened, one line each, so they happen once |
| `docs/RESULTS.md` | Measured numbers, with the conditions they were measured under |
| `docs/CODE_HEALTH.md` | The structural audit and the cleanup plan |
| `AGENTS.md` | This file — how the repository is built |

A finding from a debugging session goes in `LEARNINGS.md` the same day. A number goes in
`RESULTS.md` with its conditions or it is not a result.

---

## Hygiene — the rules with teeth

This repository is public and the data is personal. Three of these have already been
violated, at real cost.

- **Never commit `data/`, `profile.jsonl`, `.env`, or any `*.duckdb`, `*.parquet`, `*.npy`
  or `*.safetensors`.** The `.gitignore` covers all of them and explains why inline.
- **Gitignore patterns for project directories are anchored with a leading slash.** A bare
  `data/` matches at any depth: it silently excluded `src/entertainer/data/` — the entire
  ingest layer — for twenty commits, and a fresh clone could not run. ruff respects
  `.gitignore`, so that code was never linted either.
- **Secrets come from `.env` via the environment, never from a literal.** `.env.example`
  documents the keys.
- **Never prune a title the user has rated.** A vote floor decides what to *offer*, not
  what to erase; pruning a rated title orphans the verdict.
- **Item ids are positional within a build.** Anything that survives a rebuild must be
  remapped through the IMDb id. On the first real rebuild this moved 179,631 of 272,403
  titles.
- **Name the columns in every `INSERT`.** Positional `INSERT INTO t SELECT ...` breaks on
  the next migration, and breaks silently when the column counts happen to match.
- **The web interface needs a token off loopback, and that is not optional.** The page
  writes to the verdict log; an unauthenticated copy on a shared network is somebody else's
  write access to a taste profile. See `web/serve.py`.

---

## Running it

```bash
uv venv && uv pip install -e ".[encode,dev]"
cp .env.example .env               # then paste a free TMDB key

ent setup                          # download, build, enrich, embed, factorise, fuse

./scripts/entertainer start        # the app on :8756, Build Studio on :8757
./scripts/entertainer status
./scripts/entertainer logs app
./scripts/entertainer restart      # after a code change
./scripts/entertainer stop
```

`ent setup` is resumable — every stage skips work already done.
