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

Measured 2026-09-21 against the built catalogue: 73,491 titles, 47.4% with
genuine MovieLens collaborative factors, 300 held-out users, 30 answered
questions each, V-optimal elicitation. Run with
`ent eval --users 300 --budget 30`.

### The table

| arm | NDCG@10 | P@10 | MAP@10 | novelty | diversity | serendipity |
|---|---|---|---|---|---|---|
| entertainer, flat prior | **0.1898** ±0.0278 | 0.2424 | 0.1216 | 10.12 | 0.395 | 0.019 |
| ridge | 0.1835 ±0.0263 | 0.2323 | 0.1173 | 10.27 | 0.400 | 0.028 |
| **entertainer** | 0.1828 ±0.0266 | 0.2283 | 0.1188 | 10.38 | 0.394 | 0.030 |
| popularity | 0.1571 ±0.0268 | 0.1919 | 0.0871 | 9.11 | 0.570 | 0.000 |
| weighted-kNN | 0.1524 ±0.0256 | 0.1919 | 0.0921 | 10.87 | 0.393 | 0.029 |
| content-centroid | 0.1343 ±0.0263 | 0.1667 | 0.0839 | 11.09 | 0.404 | 0.030 |
| quality-prior | 0.0288 ±0.0092 | 0.0414 | 0.0110 | 11.51 | 0.771 | 0.001 |
| entertainer, no negatives | 0.0000 ±0.0000 | 0.0000 | 0.0000 | 15.38 | 0.663 | 0.000 |

Paired bootstrap of the full engine against each arm:

```
vs entertainer-no-negatives Δ=+0.1828  p=0.0000  significant
vs quality-prior            Δ=+0.1540  p=0.0000  significant
vs content-centroid         Δ=+0.0485  p=0.0000  significant
vs weighted-kNN             Δ=+0.0304  p=0.0021  significant
vs popularity               Δ=+0.0257  p=0.0635  not significant
vs ridge                    Δ=-0.0007  p=0.5417  not significant
vs entertainer-flat-prior   Δ=-0.0070  p=0.7587  not significant
```

### What this shows

**The engine is level with ridge, and ahead of everything else.** Against
plain ridge regression on identical features it is indistinguishable
(Δ=−0.0007, p=0.54) — ridge remains nominally ahead by a margin far inside
the noise. That much has not changed. What has changed is the rest of the
field: the engine now beats the rating-weighted kNN (p=0.0021) and the
content centroid (p<0.0001) significantly, where in the previous run it beat
neither. Popularity is borderline (p=0.064).

So the Bayesian treatment still has not bought anything over ridge, and on
this measurement it probably never will — the two fit the same linear model
to the same features, and the extras only matter where uncertainty is
actionable, which a static offline replay never tests. What it has bought is
a decisive margin over the simpler recommenders.

**The population prior is worth less than nothing here** (Δ=−0.0070,
p=0.76). The flat-prior variant is the top row of the table. Two consecutive
runs now put it at or below the engine that omits it, and the honest reading
is no longer "unlucky": on real held-out users the prior does not pay, and
the synthetic n=4 gain of +0.46 correlation measured a world that was too
easy.

**Implicit negatives are the whole ballgame** (Δ=+0.1828, p<0.0001). Without
sampled unrated titles the model now scores exactly 0.0000 — not one relevant
title in the top ten for any of 300 users, at the highest novelty of any arm
(15.38). That is not a weak recommender, it is a degenerate one: with no
example of "not for me", the fit collapses and the ranking surfaces the most
obscure rows in the catalogue. This remains the one component whose value is
beyond argument.

**Regularisation was the fix.** Every arm improved over the previous run, but
the Bayesian ones improved most — the engine +0.022, the flat-prior variant
+0.027, ridge only +0.011 — which is what closed a deficit that had looked
like a defeat. Pinning the effective ridge penalty rather than inferring it
from the evidence is the whole of that change; the empirical-Bayes estimate
was under-regularising roughly fourfold because the 1,000 pseudo-negatives
are a constant block and trivially fit.

### What this does not show

This is a replay of strangers from a public dataset. It says the *method* is
sound-but-unremarkable; it says nothing about whether the engine is good for
the person who built it. MovieLens held-out positives are popularity-biased —
people rate films they have heard of — which is precisely the regime where a
popularity baseline is hardest to beat and personalisation is worth least.

The measurement that matters is `ent audit` on a real verdict log, and it
cannot run until there is one.

### Prior runs

Kept deliberately, because two of them were wrong in instructive ways.

| run | engine NDCG@10 | what it actually measured |
|---|---|---|
| first | 0.0036 | a degenerate fit — the compressed reward band collapsed every weight to zero |
| second | 0.1623 | real, but against a broken kNN (0.003) and a ridge arm denied the sampled negatives |
| third | 0.1607 | all arms given the same advantages; lost to ridge |
| fourth | 0.1828 | the table above, after the regularisation fix; level with ridge |

The second run appeared to show the engine significantly beating ridge and
the centroid. It did not; it showed two handicapped baselines. Fixing them
erased the win. Retaining these rows is the point of the exercise — the
flattering run is the one that would have been quoted.
