# Ideas

Things that could make the engine better, parked here until there is evidence worth acting on.

**This is not the backlog.** [`BACKLOG.md`](BACKLOG.md) is work that has been decided on and
ordered. This file is the stage before that: findings, hypotheses and design sketches that are
worth keeping but are not scheduled. An entry graduates to the backlog when someone decides to
build it, and is deleted from here when it does.

Every entry carries what is actually known versus what is guessed, because the failure this
project keeps having is believing a good-sounding idea before measuring it. See
[`LEARNINGS.md`](LEARNINGS.md) — three separate things that improved on synthetic data did
nothing or harmed on held-out users.

| # | Idea | State |
| --- | --- | --- |
| 1 | ~~The engine only ever shows good films~~ — mostly not a problem, see the decision | **Closed**, but it exposed idea 3 |
| 3 | 58% of the model's training signal says "obscure = not for me" | **Measured** — the real finding |
| 2 | Give craft its own block in the item space | **Scoped** — see BACKLOG item 13 |

---

## 1. The engine only ever shows good films

Raised 25 September 2026 from using the app: *"the system always shows good films only, so the
chances of me rating them low is actually low — either I have watched and liked the film, or I
have not seen it. Either way I cannot rate it."*

Measured, and it holds. Sharply.

### What the numbers say

| | titles | avg IMDb rating | median IMDb votes |
| --- | ---: | ---: | ---: |
| The catalogue | 73,541 | 6.40 | 2,582 |
| What the app has actually shown | 275 | **7.86** | **203,435** |

The median title shown has **79 times** the votes of the median title in the catalogue.

By IMDb rating band, as a share of each population:

| band | catalogue | shown |
| --- | ---: | ---: |
| below 6.0 | 15.2% | **1.1%** |
| 6.0 – 6.9 | 35.1% | 13.8% |
| 7.0 – 7.9 | 27.8% | 32.4% |
| 8.0 and above | 6.9% | **52.7%** |

Three titles out of 275 scored below 6.0. More than half were 8.0 or above.

### The second half of the finding, which is the sharper one

Even the verdicts that *are* negative are negative about **well-regarded films**:

| verdict | n | avg IMDb rating of the title | median votes |
| --- | ---: | ---: | ---: |
| hated | 22 | 7.11 | 14,157 |
| disliked | 27 | 7.89 | 27,413 |
| liked | 106 | 7.10 | 46,940 |
| loved | 60 | 7.35 | 87,539 |

The films rated *disliked* have a **higher** average IMDb rating than the films rated *loved*.

So the model has never seen a bad film. Its entire notion of "not for me" is built from
acclaimed films that happened not to land — which is a real and useful signal, but it is a
different thing from badness, and the model currently cannot tell the two apart.

### Why it happens — three separate causes

1. **The Rate feed sorts by quality.** `web/feed.py` ends its query `ORDER BY quality DESC`,
   so within each language the page serves the best-regarded titles first. It was built that
   way deliberately: the fastest way to collect verdicts is to show titles the person
   recognises, and recognition correlates with acclaim.
2. **Recommendations optimise for predicted enjoyment.** Obviously and correctly — a
   recommender that showed bad films to collect labels would be a worse product. But it means
   the recommendation surface can never be the place negatives come from.
3. **The catalogue is pre-filtered.** The per-language vote floor exists to keep the catalogue
   meaningful, and it removes most genuinely obscure and bad titles before any screen sees
   them.

Each is defensible alone. Together they mean the training data can only ever be drawn from the
top of the distribution.

### Why it matters

`LEARNINGS.md` already records that a preference model trained only on rated things "has never
seen *not for me*", and that sampled negatives were worth +0.157 NDCG — the single largest
measured effect in the project. Those sampled negatives are **1,000 random unrated titles
assumed to be dislikes**. They are a synthetic stand-in for exactly the signal this finding
says cannot be collected.

That reframes them. They are not a clever trick; they are load-bearing compensation for a
structural hole in the data collection. And it may explain why the model's score accuracy
still loses to a running average while its *ranking* is now significantly real: it can order
things it has seen, but it has no calibrated sense of the bottom of the scale because nothing
at the bottom has ever been shown to it.

### Decision, 25 September 2026: the target is "for me", not "good"

Asked directly, and answered directly: *"the goal is to find films that are for me."*

**That closes most of this finding.** The engine is not trying to predict whether a film is
bad, and nobody needs help avoiding obviously bad films. So:

- The Rate feed sorting by quality is **correct**, not a bug. Recognition is what makes rating
  fast, and acclaim is the best available proxy for recognition.
- Recommendations optimising for predicted enjoyment are **correct**.
- Verdicts of "disliked" landing on titles averaging IMDb 7.89 are **not a data flaw**. They
  are the single most valuable kind of label this system can receive: an acclaimed film that
  did not land is exactly what distinguishes this person from critical consensus, and it is
  unobtainable from any public dataset.

The observation was right and the numbers hold. What it implies is much narrower than it
first appeared — and pursuing "collect some bad films" would have been building the wrong
thing for a plausible-sounding reason.

One consequence survives the decision, and it is larger than the finding that produced it.
It is recorded separately as idea 3.

### Responses considered and dropped

Kept so that nobody re-proposes them without reading the decision above.

- ~~A deliberate "is this bad?" mode.~~ Collects the wrong target.
- ~~Salt the Rate feed with low-quality titles.~~ Makes the Rate page worse at its job to
  collect labels the model does not need.
- **Ask about films already abandoned** — still worth doing, for a different reason than the
  one it was proposed for. Something started and not finished is a negative the person already
  holds an opinion about, and it is an *acclaimed-but-not-for-me* signal rather than a badness
  signal, which is the kind that matters.

- **A deliberate "is this bad?" mode.** A separate surface that samples *down* the quality
  distribution on purpose and asks only "seen it?" and "was it any good?". Honest about what it
  is for, so it does not degrade the recommendation product. Cheapest to try.
- **Salt the Rate feed.** Mix a small fraction of low-quality titles into the existing grid.
  Less honest — it makes the Rate page worse at its stated job — but needs no new screen.
- **Mine the existing negatives harder.** "Acclaimed but not for you" may be a *better* signal
  than badness for a personal recommender, since nobody needs help avoiding obviously bad
  films. If so, the right move is not to collect badness at all but to stop treating the
  sampled negatives as equivalent to stated ones.
- **Ask about films already abandoned.** Something started and not finished is a negative the
  person already has an opinion about, and streaming history sometimes records it.

---

## 2. Give craft its own block in the item space

Scoped 25 September 2026. The design, the measurements that ruled out reviews as a source, and
the four open design questions are all in [`BACKLOG.md`](BACKLOG.md) item 13 — they are kept
there rather than duplicated here because the scoping is long and the backlog is where the work
would be picked up from.

One-line version: every title's numbers are currently made from one blob of text in which a
900-character plot summary drowns out a one-line credits list, so a person whose taste runs on
execution rather than premise has no way to express that. Splitting crew into its own block
makes it separable. The data is free — TMDB returns full credits on the enrichment call
already being made.

**Related to idea 1.** Both are the same underlying complaint: the model is given a narrow,
pre-selected view of each film and of the catalogue, and then asked to learn something
general from it.


---

## 3. Most of what the model is trained on says "obscure means not for me"

Found 25 September 2026 while closing idea 1, and it is the more serious of the two.

`engine._add_sampled_negatives` draws 1,000 unrated titles and labels them weak dislikes. That
mechanism is the single largest measured effect in the project — `LEARNINGS.md` records it as
worth +0.157 NDCG, and without it the engine scores a degenerate 0.0000. It is not in doubt.

What is in doubt is **what those negatives say**, because they are drawn uniformly:

```python
rows = rng.choice(total, size=count, replace=False)
```

A uniform draw from this catalogue is not a neutral sample of "films you have not rated". It is
a sample of the catalogue's centre of mass, which is obscure and middling:

| | avg IMDb rating | median IMDb votes | share below 6.0 |
| --- | ---: | ---: | ---: |
| A uniform draw — what the 1,000 negatives are | 6.40 | 2,582 | 30.2% |
| The verdicts actually recorded as negative | 7.54 | 17,130 | ~0% |

So the two kinds of negative say opposite things. The 49 real ones say *"acclaimed, well
known, still not for me"*. The 1,000 synthetic ones say *"obscure and middling, therefore not
for me"*.

And the synthetic ones dominate:

| | count | weight | effective |
| --- | ---: | ---: | ---: |
| Real verdicts | 215 | 1.0 | 215 |
| Sampled negatives | 1,000 | 0.3 | **300** |

**58% of the model's training signal is a synthetic claim that randomly chosen obscure films
are not for this person.** Which, given the decision in idea 1, is very close to the opposite
of the target.

### Why this is a plausible explanation for the current standing

`ent audit` on 209 verdicts reports ranking that is now significantly real — Spearman +0.236 at
p=0.0007, up from +0.130 at p=0.08 — while mean absolute error still loses to a running
average, 1.91 against 1.86.

That pattern is what a model trained mostly on "obscure = bad" would produce. Obscurity is
genuinely correlated with what this person will not watch, so it orders the catalogue usefully.
But it is not what makes a specific acclaimed film land or not, so the predicted *value* is
poorly calibrated. The model has learned a filter, not a taste.

This is a hypothesis with a mechanism and numbers behind it. It is not established, and per
this project's own record it must not be believed before the benchmark agrees.

### What to try, in order

1. **Sample negatives from the same region the real verdicts live in.** Draw the 1,000 from
   titles above a popularity or quality floor rather than uniformly, so the synthetic claim
   becomes "well-regarded and well-known, still not for you" — the same claim the real
   negatives make. One line in `_add_sampled_negatives`, and it is a benchmark arm like
   everything else.
2. **Weight by how surprising the negative is.** A title the model already predicts low
   teaches nothing by being labelled low.
3. **Revisit NEGATIVE_WEIGHT once 1 is answered.** The current 0.3 was tuned against uniform
   negatives; if the draw changes, the weight that suited it probably does not.

Cheap to test, and the arms exist. It goes to the backlog only if `ent eval` shows something.
