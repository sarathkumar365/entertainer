# Backlog

Captured from a live walkthrough of the running app on 23 September 2026, then extended with
what measuring that day's full build turned up. Items 10 to 14 were added on 25 September 2026
from two further sources: running `ent audit` against the real event log rather than the
MovieLens replay, and a restatement of what the platform is for — now in the README as
"Part one — the engine" and "Part two — the agent".

**Ordered by the order of work, not by the order the pages were visited.** Start at the top.

A rendered version lives at
<https://claude.ai/code/artifact/8d510bb5-0de6-4127-a218-6c796a60f95d>.

Items 1 to 9 are interface problems: the recommendations themselves are good, and almost every
complaint is that the screen never says what it is showing you. Items 10 to 14 are not.

Items 10 to 12 are about whether the engine can be shown to work for the person using it,
which as of 25 September 2026 it cannot — not because it failed, but because the measurement
is unreadable and the log is too thin to read. Item 13 is the project's actual bet: getting
good recommendations out of very few ratings. Item 14 is what the whole thing is for.

| # | Work | Size |
| --- | --- | --- |
| 1 | ~~Taste page, phase one — two sub-tabs, figure as centrepiece, readings reworded~~ | **Done** |
| 2 | **Will I like it?** — name any title, new releases included, get a prediction | **Web done** 24 Sep; `ent why` fallback left |
| 3 | Taste page, phase two — the real map | One new build stage, plus a Canvas/WebGL render layer |
| 4 | The rest of the unexplained labels | Copy only |
| 5 | The "close to" claim | Small, but a trust problem |
| 6 | ~~The benchmark picks different users each run~~ — and replayed users it trained on | **Done**; every earlier number needs re-running |
| 7 | Build pipeline — downloads, then TMDB caching | Biggest time savings in the project |
| 8 | Library — grid view, legible sizes, verify Saved | Mechanical |
| 9 | Release publish / pull — switch it on | Configuration only |
| 10 | **Unblock the two measurements that answer "is it learning me"** | Small, and nothing above it moves this |
| 11 | Feed it properly — volume, real dislikes, and rating where it counts | No code; a habit and one nudge in the UI |
| 12 | ~~Cut the population prior~~ — the Bayesian-vs-ridge call stays open | **Prior removed** 25 Sep; ridge call blocked on item 11 |
| 13 | Work from very few ratings — the research directions | The core bet; unscoped |
| 14 | The agent — new releases, judged, acquired, ready to watch | Part two of the README; nothing built |

Reference, not work: [Working — do not touch](#working--do-not-touch) at the end.

---

## 1. Taste page, phase one

> This is a page where we should start working on.

> Too much data. I don't know what this data means. But I like the figure approach — that
> should stay, that is exactly what I wanted.

No new maths. Take what exists and restage it.

- **Two sub-tabs.** First: the figure alone. Second: everything on the page today — axes,
  readings, numbers — reworded so they can be understood.
- **Figure goes big.** Full screen, or at least the whole centre. Not a small panel above a list.
- **Keep it alive.** Particles drifting and moving. The living quality is the part he liked.
- **Label the geometry.** The axis and the two circles depict something he cannot name.
  Whatever they represent should be written on them.

Three of the undefined terms belong to this page and are part of the rewording job here, not
item 4:

| Term | Note |
| --- | --- |
| Axis 0, Axis 1, Axis 4 | Meaning unclear. Also: only one axis is visible in the figure, but several are listed in the text. |
| Surface preferences | Meaning unclear. Source: `discover.surface_preferences`. |
| Consensus quality | Shown as his highest one, no idea what it measures. One of five named side features in `features.SIDE_FEATURE_NAMES`. |

---

## 2. Will I like it?

> Let's say there's a new movie released and I want to know — is it close to my taste? I give
> a movie name and it figures out if I will like that movie or not.

Added 24 September 2026 as the immediate next task.

**Status, 24 September 2026: the web half is done.** Home has a "Will I like it?" action. The
organism flattens into a search box, lists matches when a name is ambiguous, and answers
with a score, the chance you like it and the 90% range. It was built differently from the
plan below, on the user's call:

- **Asking writes nothing.** `POST /api/judge` scores a catalogue title directly. A TMDB-only
  title is encoded, projected and scored in memory; it does not go through `/api/add` and
  never joins the catalogue. That settles the "joins the catalogue for good" constraint below.
- Side features for such a title use the catalogue's own scaling (`FeatureSpace.side_for`).
  Quality comes from the TMDB rating, exactly as `/api/add` would set it; IMDb votes are
  unknown and sit at neutral. The answer says it was judged on its description and TMDB
  rating.
- `/api/predict` and `/api/judge` now return `not_enough_evidence` with fewer than three
  verdicts.

**Left:** `ent why` falling back to TMDB, and the "close to" titles on the answer (still
blocked on item 5).

**Nothing does this end to end today.** Every piece exists; none of them are joined:

| Piece | What it does | Gap |
| --- | --- | --- |
| `ent why <title>` | Scores one title and names the liked titles it sits near | Terminal only, and only for titles already in the catalogue |
| `GET /api/predict/{item_id}` | Score out of ten, 90% interval, probability you like it | Catalogue titles only; no screen calls it |
| `GET /api/search` | Finds a name in the catalogue, then on TMDB | — |
| `POST /api/add` | Pulls a TMDB title into the catalogue and places it in the item space; the verdict is optional | Only the Rate page uses it, and always with a verdict |

A new release is exactly the case that falls through: it is on TMDB but not in the
catalogue, so `why` and `predict` both refuse it until something adds it.

**The work.**

- **A "Will I like it?" box.** Type a name and pick a hit from `/api/search`. If the hit is
  TMDB-only, `POST /api/add` with no verdict, so asking never counts as rating. Then
  `GET /api/predict` and show the answer.
- **The answer reuses the recommendation card.** Show the score and its distribution curve,
  a plain yes, maybe or no from `like_probability`, and the liked titles it sits near. The
  "close to" chips share item 5's weakness, so they must not claim more than item 5 allows.
- **`ent why` falls back to TMDB** through the same add path when the name is not in the
  catalogue.
- **Say what a new release is judged on.** A brand-new title has no MovieLens history, so its
  place in the space comes from its text alone: synopsis, genres, people. The card should
  say so, and expect a wider curve than for an established title.

**Constraints.**

- Adding a title embeds it, which needs the 1.2 GB encoder. On a machine without the item
  space, `/api/add` records the title but skips the embedding, and `/api/predict` then fails
  with "not in the current item space". That machine needs a message saying where the
  answer can be had, not a bare 409.
- An asked-about title joins the catalogue for good and can surface in recommendations
  later. That is what `add` already does and seems right, but it should be a decision rather
  than a side effect nobody chose.
- With fewer than three verdicts, `/api/predict` answers a bare 409 from a `ValueError`.
  Give it the `not_enough_evidence` code, as `/api/taste` now has, so the box can say "rate a
  few titles first" without matching on English.

---

## 3. Taste page, phase two — the real map

**Clusters by taste.** Like a spider web or a neural net: films of the same taste occupy the
same region. **Hover to identify** — hovering a region brightens or pulses it and names it
("horror", or whatever it turns out to be).

Today's figure is deliberately **not** a map. `Taste.jsx` scatters its dots from a fixed random
seed, and says why: the engine has no 2-D layout of the catalogue, and inventing one that
*looked* meaningful would be a lie. It conveys "the catalogue is vast and you occupy a corner of
it", nothing more.

The vectors needed already exist from the build — verified after the 23 September build:

| Artefact | Shape |
| --- | --- |
| `data/embeddings/content.npy` | 73,541 × 256, from the item text |
| `data/embeddings/cf_factors.npy` | 55,173 × 192, from MovieLens behaviour |
| `data/embeddings/fused.npz` | the two combined |

Nobody needs to re-derive taste. Four pieces stand between that and the page described:

| Piece | Effort |
| --- | --- |
| **Flatten to 2-D.** 256 numbers per film become an x and a y. One offline UMAP pass over 73k points, a couple of minutes, saved as an array. | The only genuinely new step, and it is small. |
| **Name the regions.** Cluster the flattened points, then find the words distinguishing each cluster from the rest. `discover._log_odds` already does exactly this for the axes. | Mostly reuse. |
| **Draw 73k points.** The current figure is SVG with 460 circles. Browsers will not hold 73,541 SVG nodes, so the render layer moves to Canvas or WebGL. | The real front-end work. |
| **Ship the coordinates.** A new endpoint with a packed binary payload — 73k points as JSON is far too heavy. | Small. |

The "inventing a layout is a lie" objection stops applying once the projection is real, because
then the corners genuinely are somewhere. The objection was to faking coordinates, not to
having them.

### Constraints, settled

- **It is a build stage, not a helper.** Its own named stage with an entry in
  `build_events.STAGES` (today an 8-tuple: sources, catalogue, tmdb, prune, embeddings, cf,
  fusion, prior), its own artefact under `data/embeddings/`, a fingerprint so an unchanged
  catalogue skips it, and progress reported into Build Studio like every other stage.
- **It rebuilds with the catalogue.** Every build producing a new fused space produces a new map
  from it. The map is derived data and must never describe a catalogue that no longer exists.
- **It goes in the published bundle**, so a machine that pulls a build gets the map with it
  rather than computing its own.

---

## 4. The rest of the unexplained labels

Copy only — no new models, no new maths. Every item is a label, a tooltip or a sentence that
does not exist yet.

### The "exploring" badge on some cards

> Some cards have an exploring tab at the top right, some don't. I don't know what that means.

It means the engine is taking a punt. Most cards are there because the model is confident; an
`exploring` card is there because the model is *unsure* and wants to find out — it did not make
the confident top ten on its own. Rating those teaches it the most.

Set in `recommend.py` as `explored=row not in exploit_rows`; rendered by `TitleCard.jsx`.

### What the score graph is showing

> Each card has a score. Nice touch, I really liked that graph thing, but I don't know what
> that graph tells me.

The number is the predicted rating out of ten. The curve is confidence: narrow and tall means
"confident it is about this good", wide and flat means "could be anywhere in here". The `±`
value is the same thing as a number.

**The 0 / 5 / 10 ruler.** Those are ticks on one scale, not three zones — there is no meaning
to "0 to 5" versus "5 to 10". The hill's position is the prediction and its width is the doubt;
a hill straddling 5 means "might be middling, might be good". If that reads as two regions,
name the two ends rather than adding zones.

**Decided: leave the card alone otherwise.** Decluttering was considered and dropped.

### Save for later

Does not know what it does or where it goes. Library labels it "kept out of recommendations,
but not a verdict" — that string exists but never reaches the button that creates one.

### The Evidence page

> Honestly, I don't understand anything.

Not a layout problem — the charts are good and stay as they are. It is the words around them.

- **The opening sentence.** "Learning from you — every prediction below was made…" should be
  natural language, not a technical statement.
- **Mean absolute error**, **running average**, **error first vs last** — each needs a
  plain-language equivalent.
- **Off-policy check.** "10 of 30 recommendations have an outcome" does not parse, nor does the
  note that rating from the Recommend tab is what produces one.
- **Blind test.** Same.
- **Verdicts used** — understood. The one term that reads fine.

---

## 5. The "close to" claim is not believable

Not a missing label. The app makes a claim a viewer can check, he checked it, and it did not
hold up. That costs trust in a way a mystery label does not.

**Kartikeya 2** was offered as close to **Ennu Ninte Moideen** (biographical romantic drama),
**Drishyam 2** (crime / psychological thriller) and **The Witch: Part 2** (supernatural
sci-fi). He has seen all three and not the recommendation, so the chips are his only evidence —
and on genre they share nothing.

He is explicit that he is *not* questioning the pick. He wants to know how it was made and how
the closeness claim is justified.

Underneath: "close to" is nearness in the fused taste space — part text, part "people who watch
these also watch that" — not genre similarity. Two films with nothing in common on paper can
sit near each other there, and that is what makes the engine useful. But the chip is labelled as
though it were obvious resemblance, and the bar is low: anything above `min_sim = 0.15` in
`discover.nearest_liked` qualifies, so weak matches print with the same confidence as strong
ones.

---

## 6. The benchmark picks different users each run

**Fixed.** `simulate.load_user_histories` drew its simulated users with a seeded
`rng.choice` over the output of `group_by("userId")`. Polars does not keep order in a
`group_by`, so the same seed picked a different 300 users on every run. The popularity arm
shows it: it has no randomness of its own, yet it scored NDCG@10 0.1571 in
`reports/bench-v4.log` and 0.1447 in `reports/bench-v5-neg100.log`.

The candidates are now sorted before the draw, and the returned histories are sorted by user
id. The order matters as much as the set: users take turns drawing from one shared generator,
so the same users in a different order still get different splits.

**Also fixed: the replayed users were not the held-out ones.** `ent data cf` withholds
2,000 MovieLens users from the CF factors and the prior, and `ent eval` checked that record
existed but never drew from it. Nearly every replayed user had helped train the fused space
every arm ranks in, so the absolute scores were inflated. The check was also skipped
entirely whenever no prior was fitted. `integrity.benchmark_users()` now returns the
holdout or refuses, and `load_user_histories` requires the eligible users as an argument.

**Consequence.** No benchmark number from before these fixes stands. Comparisons between
two runs were invalid because they scored different users. Absolute scores were inflated by
the leak, and that affects the arms unevenly, so even the gaps between arms in one run are
suspect. Re-run from scratch before drawing conclusions.

---

## 7. Build pipeline

Measured on the 23 September build — **2 h 43 m** wall clock:

| Stage | Time |
| --- | --- |
| sources | 2,309 s — 38.5 min |
| catalogue | 27.9 s |
| tmdb | 4,634 s — 77.2 min |
| prune | 5.8 s |
| embeddings | 2,779 s — 46.3 min |
| cf | 81.1 s (overlapped with tmdb) |
| fusion | 5.1 s |
| prior | 8.6 s |

### Downloads: split each file across connections

IMDb's CDN throttles **per connection**, not per IP. Measured: 1 connection 0.33 MiB/s,
6 → 1.91, 12 → 3.74, 16 → 4.80. Linear, per-stream speed unchanged. The link is not the limit.

`download.fetch` opens one connection per file, and the files run 8.6 MB to 784 MB
(`title.principals.tsv.gz` is 747 MiB), so the stage ends when the largest file finishes alone:
747 MiB ÷ 0.33 MiB/s = 37.7 min, against 38.5 min measured for the whole stage. Five workers
idle through the last half hour.

Chunk each file into byte ranges across ~16 connections — the `Range` machinery already exists
in `fetch` for resume. **38.5 min becomes about 7.**

`ac7242d` already added file-level concurrency; what is missing is splitting one file.

### TMDB: stop re-buying enrichment the prune throws away

77 minutes of API calls, and rate is not the lever — TMDB returned 429s when offered roughly 70
requests a second, and the build already runs at 39 against a self-imposed 40/s cap. Request
count is the lever.

`prune_by_language` deletes rows outright, so the TMDB data bought for pruned titles dies with
them. The 23 September build enriched 131,140 titles; the catalogue held 272,580 rows before the
prune and 73,541 after. Everything deleted will be re-added empty by the next rebuild from a
fresh dump, and paid for again.

Keep enrichment in its own table keyed on `imdb_id` that the prune never touches. First build
still pays in full; later builds fetch only genuinely new IMDb ids. **Hours become minutes.**

*Rejected:* raising the lowest vote floor above 100 would also cut requests, but it thins
Kannada and Malayalam coverage, which is the point of the per-language scheme.

### Two smaller things in the same stage

- The batch write blocks the event loop: `apply_enrichment` takes 175–320 ms per 500 rows and
  runs inside the async handler, freezing all 40 workers each time. Move it to a thread and
  raise `batch_size` from 500 to 2,000 — measured 2,880 rows/s at 500 versus 5,100 at 2,000.
- `enrich_async` accumulates every result in a list its caller discards — a few hundred MB held
  for nothing while the encoder wants RAM.

---

## 8. Library

- **Needs a grid view, not a list.** "The grid can be smaller."
- **Everything is too small to read.** "Not bad, but everything is very small. I can't read
  anything."
- **Saved section looked empty, then worked.** Of Saved / Watched / Rated, only Rated had
  anything. Save for later appeared not to land at first; on retry it did. Reproduce before
  treating it as a bug.

---

## 9. Release publish and pull — switch it on

The code landed in `df5f83d` and has never been used.

`ent release publish` exports the catalogue, fused space, population prior and content encodings
and attaches them to a GitHub release on a private builds repository. `ent pull` fetches the
newest one and imports it, verifying every file against the manifest and remapping the local
event log onto the new item ids in one transaction. Personal data is never touched.

So a second person needs no GPU and no 2 h 43 m build — they pull the artefacts and start
rating.

What is missing, verified:

- `ENTERTAINER_RELEASES_REPO` is **absent from `.env`** (it is present but blank in
  `.env.example`).
- The private repository it points at does not exist yet.
- `gh` **is** installed and logged in as `sarathkumar365`, so that half is already satisfied.
  Publish refuses unless the repository is private — the data is IMDb- and TMDB-derived and must
  not be redistributed.

**Limit worth knowing now.** The `events` table has no user column, so one install is one
person's taste. Handing the system to somebody else means giving them their own install, which
is what pull is for. Several people sharing one server would need a schema change, and nothing
today is built for it.

How the personal model trains is documented in
[HOW_IT_WORKS.md](HOW_IT_WORKS.md#two-kinds-of-learning).

---

## 10. Unblock the two measurements that answer "is it learning me"

Added 25 September 2026, from running `ent audit` against the live event log rather than
against the MovieLens replay.

Every other item on this list is copy, layout or build speed. None of them move the only
question the project exists to answer. This one does, and it is small.

### What `ent audit` says today

186 verdicts, 25 September 2026:

| measure | value | reading |
| --- | --- | --- |
| mean absolute error | 1.84 / 10 | lower is better |
| vs running-average baseline | 1.79 / 10 | **model loses** |
| error: first vs last | 0.00 → 1.95 | flat or worse |
| learning slope | +0.059 per 100 verdicts | p=0.039 |
| 90% interval coverage | 0.87 | calibrated |
| rank correlation | +0.130 | p=0.0806 |

Read at face value that says the engine gets *worse* as it learns, significantly. It does not
say that, for the reason below — but the MAE line does not depend on ordering and does stand:
on this log the model does not beat predicting the user's own average.

### The confound: the log is two instruments glued end to end

`events` holds two populations that do not interleave at all.

| source | n | timestamp range | values present |
| --- | --- | --- | --- |
| `netflix` | 84 | 2025-06-19 → 2026-09-18 | `1.0`, `7.5`, `10.0` |
| `web` | 108 | 2026-09-21 → 2026-09-25 | `1.0`, `4.0`, `7.5`, `10.0` |

**Settled, 25 September 2026: a Netflix verdict is a web verdict.** The import exists so the
user does not have to re-click ratings he has already given; the intent was always "treat
these as though I clicked them here". So `source` is provenance only. Nothing downstream may
branch on it, and there is no second pipeline to build or reconcile.

That resolves the framing but not the bug. The real cause is narrower, and confirmed by
reading the log:

```
first 20 verdicts, in ts order:
7.5 7.5 7.5 7.5 7.5 7.5 7.5 7.5 7.5 7.5
7.5 7.5 7.5 7.5 7.5 7.5 7.5 10.0 7.5 7.5
```

The first seventeen verdicts all carry the same value. `prequential.run` starts predicting at
`MIN_TRAIN = 5`, so the first ten predictions are made by a model fitted on a constant, asked
to predict that same constant. It scores exactly `0.00` — not accuracy, an absence of variance
to be wrong about.

`trend()` then compares that window against the last ten, which do have spread, and
`learning_slope()` regresses raw absolute error against step count across the whole run. Both
therefore measure **how varied the labels happened to be at each point in the log**, and only
incidentally the model. The audit code itself was verified and is honest — it refits on
everything before each verdict and predicts blind, with no leakage. The inputs are what make
the trend meaningless.

### The work

- **Measure skill, not error.** Report the trend on `baseline_error - model_error` rather than
  on `absolute_error`. The running-average control faces exactly the same label variance at
  every step, so the difference is immune to the problem above while a raw error curve is not.
  Touches `learning_slope()` and `trend()` in `evaluation/prequential.py`.
- **Refuse to report a trend on a degenerate window.** If a window has near-zero variance in
  `actual`, the error over it carries no information; say so instead of printing `0.00` beside
  the word "improving". Same file, plus the rendering in `commands/diagnose.py`.
- **Leave `source` alone.** It stays as provenance. Do not branch on it, do not weight by it,
  do not split the audit by it — per the decision above.
- **Unblock the off-policy estimate.** It needs 30 recommendations with an outcome and has 16.
  Only verdicts given from `/recs` produce one. Until then the measurement that would test the
  engine where its uncertainty is actually actionable cannot run at all — which is precisely
  the defence RESULTS.md offers for the Bayesian machinery being level with ridge.

### Why this matters more than its size

`docs/RESULTS.md` says the offline replay "says nothing about whether the engine is good for
the person who built it", and that the measurement that matters is `ent audit` on a real
verdict log. That log now exists. It is currently unreadable for the reason above, and it is
also still thin: n=186, 82% positive, with the negative examples arriving almost entirely from
one import.

That is below the threshold where the thesis can be tested at all — the README puts the point
where curvature starts paying at "a couple of hundred". So the honest standing is **not enough
evidence**, not **it failed**. Fixing the instrument split is what makes the difference
between those two readings visible as the log grows.

---

## 11. Feed it properly — and rate where it counts

Added 25 September 2026, alongside item 10. That item covers the measurement being
*unreadable*; this one covers the log being *thin*. They are separate problems and fixing
either alone leaves the question open.

### The state of the log

192 rate events on 25 September 2026:

| value | meaning | n |
| --- | --- | --- |
| `10.0` | loved | 59 |
| `7.5` | liked | 99 |
| `4.0` | disliked | 13 |
| `1.0` | hated | 21 |

**82% positive.** The model has seen a great deal of "this is for me" and very little of the
opposite, and 10 of the 21 hated verdicts arrived in a single Netflix import rather than from
deliberate rating. A preference model learns a boundary; it is currently being shown one side
of it.

This is the same shape as the finding in LEARNINGS.md — "a preference model trained only on
things the user rated has never seen *not for me*" — which is why sampled negatives exist and
why they are worth +0.1828 NDCG. Sampled negatives are a stand-in for real ones. Real ones are
better.

### Where a verdict is given changes what it is worth

This is not obvious from any screen, and it is the single least-known fact about the system:

| Given from | Trains the model | Produces an off-policy outcome |
| --- | --- | --- |
| `/recs` | yes | **yes** |
| `/rate` | yes | no |
| `ent loved` etc. | yes | no |
| Netflix import | yes | no |

Only `/recs` logs the propensity a title had of being shown, so only a verdict given there can
answer "were those slates any good?". The off-policy estimate needs 30 such outcomes and has
**16**. Two hundred verdicts given on `/rate` move that counter by zero.

`README.md` states this correctly under "The interface". Nothing in the running app does.

### The work

- **Say it in the app.** `/recs` should carry one line making the point — rating here is worth
  more than rating elsewhere, because it is the only place that measures whether the picks
  were good. Copy only.
- **Show the counter.** The Evidence page already says "16 of 30 recommendations have an
  outcome" and item 4 flags that string as unreadable. Rewriting it is the natural place to
  say what closes the gap.
- **Rate 14 or more from `/recs`.** Not code. It is the smallest action in this document with
  the largest unlock: it is what lets the off-policy check run at all, and that check is the
  only measurement that tests the model where its uncertainty is actionable.
- **Then bulk toward roughly 400 verdicts, leaning on dislikes.** `ent bulk` takes a file of
  `title | verdict` lines, which is the fastest route for titles already known to be
  disliked. The target is a less lopsided log, not a bigger one.

### What this does not fix

Worth stating so the items are not conflated. More verdicts do **not** repair item 10's
instrument split — that is a code change, and more data buries the confound rather than
dissolving it. They do **not** touch item 5's "close to" claim, which is a threshold and a
label. They will not move the offline benchmark in `docs/RESULTS.md` at all, since that
replays MovieLens strangers and never sees this log.

---

## 12. Decide the fate of the parts that are not paying

Added 25 September 2026. Recorded here because `docs/RESULTS.md` measures these honestly but
nothing acts on the measurements, and an unowned negative result quietly becomes permanent.

Two components have now been measured as not earning their place:

| Component | Standing | Runs |
| --- | --- | --- |
| Population prior | Δ=−0.0070, p=0.76 against the flat-prior arm — the flat-prior variant is the top row of the results table | Measured below or level twice consecutively |
| Bayesian treatment vs plain ridge | Δ=−0.0007, p=0.54 on identical features | Four runs, never ahead |

RESULTS.md gives the correct defence for the second: the extras "only matter where uncertainty
is actionable, which a static offline replay never tests". That defence is sound and it is also
**currently untestable**, because the measurement that would test it is the off-policy estimate
blocked in item 11. The two items are linked: item 11 unblocks the evidence this decision
needs.

The first has no such defence. The population prior was justified by a synthetic n=4 result
(+0.46 correlation) that RESULTS.md now describes as having "measured a world that was too
easy", and it has been at or below the flat-prior arm on real held-out users twice.

### Decision, 25 September 2026

**Cut the population prior.** Two consecutive held-out runs put it at or below the arm that
omits it, and its original justification — a synthetic n=4 result of +0.46 correlation — is
described in RESULTS.md as having "measured a world that was too easy". There is no evidence
left supporting it.

**Done, 25 September 2026.** Removed across fourteen files: `models/population.py` and its
tests deleted, the `prior` build stage gone so a full build is seven stages rather than eight,
`ent data prior` gone, the block-diagonal reparameterisation and its `alpha_anchor` hyperprior
out of `models/taste.py`, `Engine.prior()` gone, the `entertainer-flat-prior` benchmark arm
gone along with `simulate._ACTIVE_PRIOR`, `population_prior.npz` out of the bundle's `SPACE`
and out of the build manifest, and README section 3b deleted.

**The published bundle still imports.** `build-20260925-1806` was already on the releases
repository, and its manifest lists `population_prior.npz` with a checksum. `bundle.verify`
walks the *manifest's* checksum list rather than this code's `SPACE` tuple, so the file is
still verified on the way in; the restore loop then simply does not copy it out. No re-publish
is needed, and a regression test in `tests/test_bundle.py` rebuilds exactly that archive shape
and asserts it restores.

Keep the *evidence* in `docs/RESULTS.md`. A component removed for a measured reason is a
result, and deleting the reason alongside the code is how a project re-adds the same idea two
years later.

### Still open: the Bayesian treatment vs plain ridge

**Do not decide this yet.** The defence in RESULTS.md is sound — the extras only matter where
uncertainty is actionable, which a static offline replay never tests — and it is currently
untestable because the off-policy check is blocked in item 11.

There is now a second reason to keep it regardless of that result: **the agent in Part two of
the README depends on it.** An agent that acquires films unsupervised needs to distinguish
"he will like this" from "I have no idea", and a point estimate cannot. Interval coverage
measures at 0.87 against a nominal 0.90 on the real log, so the mechanism demonstrably works
even though accuracy is level with ridge. Ridge cannot supply that at all.

Revisit once the off-policy estimate runs.

---

## 13. Work from very few ratings — the research directions

Added 25 September 2026. This is the project's defining constraint, restated in the README as
"a system that only becomes good at five hundred verdicts has already failed, because nobody
will reach five hundred."

Everything currently built attacks this from one side: make the *model* frugal — closed-form
Bayesian regression, a population prior, information-gain question selection. That side is
close to exhausted. The three directions below attack it from the other side: make each
verdict *teach more*.

Survey done 25 September 2026; sources at the end of this section.

### Direction A — extract craft attributes with a language model

**The gap.** The engine's content tower embeds a prose card and gets back 256 numbers. Nothing
in that pipeline isolates *how a film is made* from *what it is about*, and craft is what the
README now says taste actually runs on — pacing, whether it grips in the first ten minutes,
how it is shot and cut, whether it explains itself. Thallumaala and Avatar are both cases where
the premise predicts the wrong answer and the execution predicts the right one.

**The approach.** Use an LLM to extract structured attribute-sentiment pairs from reviews and
synopses, producing an explicit per-title craft vocabulary rather than an opaque embedding.
This is a well-established line — aspect extraction for explainable recommendation — and the
recent work uses exactly this shape: induce a compact corpus-level aspect vocabulary, then
extract aspect-opinion triples against it as constraints.

**Why it fits the constraint.** A verdict currently teaches the model about one point in a
192-dimensional space. Under aspect features a verdict teaches it about *qualities*, which
transfer to every other title sharing them. That is the whole of the low-data argument.

**Bonus.** It fixes item 5 for free. "Close to Drishyam 2" is unbelievable because nearness in
a fused latent space is not legible; "both hold back information from you and pay it off late"
is checkable. The explanation stops being a claim the user cannot verify.

**Cost.** One new build stage over 73k titles, and a source of review text the catalogue does
not currently hold.

### Direction B — ask for comparisons, not ratings

**The finding.** Preference-elicitation research converges on two results that the current
cold start does not use: ask about *attributes* rather than items, and prefer *pairwise
comparisons* to numeric scales, which extract more signal per question. Completion rates fall
sharply past five to eight questions, which bounds the whole budget.

**Against the current design.** The D-optimal question selection is already optimal *given*
that the answer is a rating of one item. It has never been tested against a different question
format, and the format may matter more than the selection criterion.

**Caution.** The README's central claim is that the engine never asks you to describe yourself,
because people cannot. "Ask about attributes" sits in obvious tension with that. The
reconciliation, if there is one, is that a *comparison* is not a self-description — "this one
over that one" is still a verdict, just a denser one. Adopting attribute questions wholesale
would change what the project is.

### Direction C — keep a written taste profile beside the vector

**The approach.** Maintain a natural-language description of the user's taste, refreshed as
verdicts accumulate, and use it as context for scoring. Few-shot prompting with a handful of
liked and disliked titles is measurably better than zero-shot, and is competitive with
cold-start recommenders specifically when natural-language preferences are supplied.

**The known limit, stated plainly.** Both zero-shot and few-shot LLMs *under-perform* trained
recommenders that have interaction data. That is the consistent finding and it should not be
wished away. It is also not the regime this project operates in: at n=186 there is no
interaction data to speak of, which is precisely where the ranking reverses.

**Bonus.** It makes `/taste` say something a person can read and dispute. Today it prints
"Axis 0, Axis 1, Axis 4" and the user has no way to tell whether it is right about him.

**Caution.** This is the direction most likely to produce something that *sounds* right and is
not. Any version of it ships behind `ent audit` on the real log, or it is not shipped.

### How to decide between them

Not by argument. `docs/LEARNINGS.md` records three separate things that improved on synthetic
data and did nothing or harmed on held-out users, and two in-sample results that reversed under
holdout. The same rule applies here.

**Direction A first** — it is the only one that also repairs item 5, it produces features the
existing Bayesian model can consume without redesign, and it can be measured with the benchmark
that already exists by adding the craft features and re-running the arms. B and C both change
the interaction model or the scoring path, which is a larger commitment on weaker evidence.

**Blocked on item 10 and item 11.** None of this can be evaluated while the learning trend is
unreadable and the log is at 186 verdicts, 82% positive. Fix the measurement, then feed it,
then try Direction A.

### Sources

- [Understanding Before Recommendation: Semantic Aspect-Aware Review Exploitation via LLMs](https://dl.acm.org/doi/10.1145/3704999) — ACM TOIS
- [HADSF: Aspect Aware Semantic Control for Explainable Recommendation](https://arxiv.org/abs/2510.26994)
- [Enhancing recommender systems with LLM-extracted explicit features](https://link.springer.com/article/10.1007/s10115-025-02665-2) — Knowledge and Information Systems
- [Large Language Models as Conversational Movie Recommenders: A User Study](https://arxiv.org/abs/2404.19093)
- [Do LLMs Understand User Preferences? Evaluating LLMs On User Rating Prediction](https://arxiv.org/pdf/2305.06474)
- [Deep Rating Elicitation for New Users in Collaborative Filtering](https://arxiv.org/pdf/2402.16327)
- [Small LLMs can be good cold-start recommenders](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1705245/full) — Frontiers in AI, 2026
- [Awesome-Cold-Start-Recommendation](https://github.com/YuanchenBei/Awesome-Cold-Start-Recommendation) — maintained survey list

---

## 14. The agent — new releases, judged, acquired, ready to watch

Added 25 September 2026. Specified in the README under "Part two — the agent". **Nothing is
built.** This item exists so the engine work above is ordered against something real.

### What it needs from the engine

| Need | State |
| --- | --- |
| Score an arbitrary title, including one not in the catalogue | **Exists** — `POST /api/judge`, built for item 2 |
| A calibrated "I am not sure", so it can decline | **Exists and works** — interval coverage 0.87 against nominal 0.90 |
| Predictions good enough to act on unsupervised | **Not yet** — see item 10; the model does not beat a running average on the real log |

The middle row is why item 12 does not cut the Bayesian machinery even though it is level with
ridge on accuracy. An agent that acquires films without being asked must be able to say "I have
no idea", and a point estimate cannot.

### The pieces

- **A source of new releases.** TMDB already supplies enough to enumerate recent titles, and
  the live feed in `web/live.py` already talks to it.
- **The back catalogue too.** A 1994 film the user has never seen answers "what should I watch
  tonight" as well as a new one. Restricting the agent to new releases would be a smaller
  system than the one specified.
- **A decision rule with a threshold.** Conservative by default. A wrong acquisition costs disk
  and trust; a missed one costs nothing, because the user can still ask.
- **Delivery into Jellyfin.** The target is a library path on the home server.
- **Feedback.** Whether the title was watched, finished or ignored should become a verdict.
  This is the only ratings source that costs the user nothing, which makes it directly relevant
  to item 13's constraint.

### Open questions

- **Where titles are acquired from is not settled**, and it is the one decision that must be
  made explicitly rather than inherited from whatever library is convenient. Whatever the agent
  does must respect what the user is licensed to hold.
- **Disk budget and an eviction rule.** An unsupervised agent with neither fills the disk.
- **Sharing.** `events` has no user column, so one install is one person's taste. Several people
  on one Jellyfin server is a schema change. Noted in item 9 as well.

---

## Working — do not touch

Reference, not work. Recorded so the items above do not break it.

| Where | What |
| --- | --- |
| Home | The whole landing page. Animation, the copy tucked into every corner, the "learn for a few minutes, then see what it thinks" framing, the navigation. |
| Recommend | The picks themselves. Engine quality is not on this list. |
| Recommend | Visual design of the cards — layout, density, hierarchy. |
| Recommend | Score plus its distribution graph. The idea and the drawing are both good; only the meaning is missing. |
| Recommend | Film / series tabs. |
| Recommend | Verdict buttons. Ratings register correctly. |
| Rate | The page as a whole. Understood without help, visual hierarchy is very nice. |
| Taste | The figure approach — "exactly what I wanted". Direction right, execution needs work. |
| Evidence | How the data is drawn. The charts stay as they are. |
