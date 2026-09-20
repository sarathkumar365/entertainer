# Measurement

This document records how the engine is evaluated and what the measurements
came out at. It is written to be checkable: every number here can be
reproduced with the command printed beside it.

Results are filled in at the bottom once the full catalogue build completes.
The methodology is fixed in advance, deliberately — deciding what counts as
success after seeing the numbers is how offline recommender evaluation
usually goes wrong.

---

## 1. Why offline evaluation at all

The honest problem with building a recommender for one person is that you
cannot evaluate it on that person until months have passed. Two measurements
are therefore run instead, answering two different questions.

**Does the method work?** — measured by replaying strangers. MovieLens users
held out of collaborative-filtering training arrive as cold-start users, the
engine questions them, and the resulting ranking is scored against their
real held-out ratings.

**Is it learning *me*?** — measured prequentially on the user's own verdict
log, once there is one. Every prediction is made by a model refitted on the
verdicts that preceded it and blind to the answer.

Neither is a substitute for the other, and neither is a substitute for
actually liking the films it suggests.

---

## 2. The replay protocol

```bash
ent eval --users 300 --budget 30
```

Each simulated user's rating history is split: 60% is available for the
engine to question them about, 40% is held back for scoring. The engine asks
its questions; the user answers from their real ratings, or says "haven't
seen it" when asked about something they never rated. After a fixed budget of
*answered* questions the engine ranks the entire catalogue minus what it was
told about, and that ranking is scored against the held-out 40%.

Relevant means a MovieLens rating of 4.0 or above on the 0.5–5 scale.

### What stops this from flattering the model

- **No collaborative leakage.** Held-out users are excluded from the ALS fit
  (`ent data cf --holdout 3000`), so the item factors the engine scores with
  have never seen their opinions.
- **No prior leakage.** The same users are excluded from the population prior
  fit. `ent eval` refuses to print a number if it cannot confirm which users
  were held out.
- **No split leakage.** Elicitation and evaluation ratings are disjoint
  subsets of each user's history, asserted in `tests/test_simulate.py`.
- **Equal information.** Every arm receives byte-identical answers, so a win
  cannot come from the model having been told more than a baseline was. Also
  asserted by test.
- **The full haystack.** Candidates are the whole catalogue, not a
  pre-narrowed shortlist containing the answers.

### Arms

| arm | what it is | what it isolates |
|---|---|---|
| `popularity` | rank by IMDb vote count | the "just show them what everyone watches" floor |
| `quality-prior` | rank by shrunk, per-language-standardised quality | the "just show them good films" floor |
| `content-centroid` | cosine to the mean of liked items | the standard naive personalisation |
| `weighted-kNN` | rating-weighted item kNN in the fused space | whether the Bayesian model beats a neighbourhood method on the same representation |
| `ridge` | ridge regression on identical features | the model minus its Bayesian machinery |
| `entertainer-flat-prior` | the engine with an isotropic prior | the model minus its population knowledge |
| `entertainer` | the full engine | — |

The last three matter most. Beating `popularity` proves very little; beating
`ridge` on identical features isolates the contribution of the empirical-Bayes
treatment, and beating `entertainer-flat-prior` isolates the population prior.

### Metrics

Accuracy is reported as NDCG@10, precision@10, MAP@10 and MRR@10. Alongside
them, and not optionally: novelty (mean self-information of the slate),
intra-list diversity, and serendipity (relevant hits a popularity baseline
would not have surfaced).

A recommender optimised purely for NDCG converges on the safest, most popular,
most already-known titles. It scores well and is useless, because a
recommendation you were always going to find is worth nothing. Reporting the
diversity metrics alongside makes that failure visible rather than hidden.

Significance is by paired bootstrap over users — paired because every arm saw
the same users with the same answers, so the enormous user-to-user variance
cancels. An unpaired test here would hide almost any real effect.

---

## 3. The elicitation comparison

```bash
ent eval --users 300 --budget 30 --elicitation v-optimal
ent eval --users 300 --budget 30 --elicitation d-optimal
ent eval --users 300 --budget 30 --elicitation random
```

Three question-selection strategies over identical everything else.
`v-optimal` maximises predicted reduction in predictive variance across the
catalogue; `d-optimal` is the textbook criterion, maximising information about
the parameters; `random` is the control.

V-optimal is the shipped default because it optimises the objective the system
is judged on, but which one actually recommends better is an empirical
question. The synthetic fixtures could not settle it — everything saturates
past about twenty questions there — so it is settled here.

---

## 4. The self-improvement measurement

```bash
ent audit
```

Walks your own verdicts in chronological order, refits on everything before
each one, and predicts it blind. Reports:

- **mean absolute error**, against a running-average control. A model that
  cannot beat predicting your own average has learned nothing about you.
- **error over time**, first window versus last, plus the OLS slope of error
  against verdict count and its p-value. A significantly negative slope is the
  self-improvement claim stated as a number.
- **interval coverage** — the fraction of outcomes inside the 90% predictive
  interval, which should sit near 0.90. A model whose error bars are wrong is
  worse than one with none, because the exploration policy is built on them.
- **rank correlation** between predicted and actual preference.

Needs at least eight verdicts before it will say anything, and says so
otherwise.

`ent audit` also reports an off-policy estimate: what today's model would
have scored on the slates an older, worse model actually showed you. This is
possible only because every slate records the probability each title had of
being shown; without that, comparing verdicts collected under different
models is apples to oranges. It is reported separately and hedged, because it
genuinely is the weaker measurement — self-normalised importance weighting is
still high-variance on a few hundred samples, and it assumes the candidate set
has not shifted underneath it.

---

## 5. Results

*Pending the full catalogue build. This section will contain the tables
produced by the commands above, with the catalogue size, language
distribution and CF coverage they were measured against.*
