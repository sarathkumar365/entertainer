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
| 1 | The engine only ever shows good films, so it can never learn what bad looks like | **Measured, confirmed** — no fix chosen |
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

### Possible responses — none chosen

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

### What would settle it

The sampled-negative mechanism is already a benchmark arm, so the question is answerable with
machinery that exists: whether real low-quality negatives change anything the 1,000 synthetic
ones do not. That cannot be tested until some real ones have been collected, which is what
makes the first response above the one worth trying first.

**Open question worth resolving before any of it:** is the goal to predict *badness*, or to
predict *"not for me"*? They are different targets, this project has only ever cared about the
second, and the answer decides whether this finding needs a fix at all.

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
