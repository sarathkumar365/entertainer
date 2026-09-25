# Backlog

Captured from a live walkthrough of the running app on 23 September 2026, then extended with
what measuring that day's full build turned up.

**Ordered by the order of work, not by the order the pages were visited.** Start at the top.

A rendered version lives at
<https://claude.ai/code/artifact/8d510bb5-0de6-4127-a218-6c796a60f95d>.

The recommendations themselves are good. Almost every problem below is that the screen never
says what it is showing you.

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
