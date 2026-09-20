# entertainer

A personal recommendation engine for film and television that learns what you
like from **titles and verdicts alone** — no genres, no tags, no explaining
yourself — across English, Malayalam, Tamil, Telugu, Kannada, Korean,
Japanese, European and other catalogues.

```bash
ent loved "Kumbalangi Nights"
ent hated  "Morbius"
ent recs --lang ml,ta --movies
```

It runs entirely on your machine. Your taste data never leaves it.

---

## The problem this is actually solving

Most recommenders ask you to describe yourself: pick your genres, rate these
sliders, choose your moods. That fails because the things that decide whether
you like a film are not in any tag vocabulary. Nobody has ever liked a film
*because* it was tagged `Drama`. They liked it because of its pace, its
restraint, the way it refused to explain itself, a particular actor's
stillness — and they usually could not have told you that in advance.

So this engine never asks. It takes the only input people are reliable about —
*I loved that one, I couldn't finish that one* — and works backwards to the
structure underneath. The reasons are **discovered**, not declared.

Genres and keywords do appear in the system, but only in two places: buried
inside the text an encoder reads, where the model is free to ignore them, and
at the very end, as vocabulary for *describing* what was discovered. They
never constrain what can be learned.

---

## How it works

```
IMDb bulk dumps ─┐
MovieLens-32M  ──┼─> catalogue ─┬─> item cards ──> multilingual encoder ─┐
TMDB API       ─┘               │                                        ├─> fused
                                └─> co-consumption matrix ──> iALS ──────┘   latent
                                                                             space
                                                                               │
   your verdicts ──> Bayesian posterior over taste <──────────────────────────┘
                              │
                              ├─> Thompson-sampled recommendations
                              ├─> information-gain questions (cold start)
                              └─> post-hoc axis discovery ("what it learned")
```

### 1. The catalogue

**IMDb bulk exports** form the spine: complete, free, and — the reason they
were chosen over TMDB as the base — they carry small-industry cinema with the
same fidelity as Hollywood. A vote threshold tuned for American films would
erase most of the Malayalam catalogue, so the floor is **per-language**: 200
votes on a Malayalam film is a comparable cultural footprint to 2,000 on an
English one.

**TMDB** then enriches each title with a synopsis, a curated keyword
vocabulary, and an authoritative `original_language`. The synopsis matters
most — it is the only place where pace, tone and moral posture are written
down at all.

**MovieLens-32M** contributes 32 million ratings of co-consumption behaviour.

### 2. The item space

Two towers, fused.

The **content tower** embeds a prose card per title with
`Qwen3-Embedding-0.6B` — as of 2026 the leading open-weight multilingual
embedding family, covering 100+ languages, which is non-negotiable for a
catalogue spanning four South Indian languages plus Korean and Japanese.
Matryoshka truncation to 256 dimensions costs almost nothing in quality and
makes the full item matrix cheap to scan.

The **collaborative tower** factorises the MovieLens matrix with implicit
ALS. This captures what no synopsis states: that a certain kind of viewer
reliably crosses between two films sharing no surface feature at all.

MovieLens only covers ~87k titles. For everything else — most of the
non-English catalogue, and every recent release — collaborative factors are
**imputed from content** by a ridge map fitted on the overlap. Imputed factors
are deliberately left shrunk rather than variance-inflated, and a per-item
confidence scalar rides into the space so the model can learn how much to
trust that block. Manufacturing confidence the system does not have would have
been the easy option.

The two towers are concatenated and rotated by PCA into an orthogonal basis.
That rotation is what later makes the learned taste vector legible.

### 3. The preference model

Three constraints decided this, in order:

1. **The supervision is names and a verdict.** Nothing else. The structure has
   to be found, not supplied.
2. **There will never be much data.** A person might label 200 films in a
   year. Anything with the capacity to overfit 200 points will overfit 200
   points — which rules out a fine-tuned transformer or a deep two-tower net,
   not because they are bad ideas but because they are the wrong ideas at
   n=200. The scaling laws that make deep recommenders win start several
   orders of magnitude above one human's viewing history.
3. **The model must know what it doesn't know.** Without calibrated
   uncertainty a recommender spends its life re-recommending the middle of the
   distribution you already occupy.

The answer to all three at once is **Bayesian linear regression with
empirical-Bayes hyperparameters**, lifted through random Fourier features when
— and only when — the marginal likelihood says the data supports curvature.
Closed form. Fits in microseconds. Exact posterior covariance rather than a
dropout approximation. And its posterior mean *is* the interpretable taste
vector.

Capacity selection is itself Bayesian: candidate feature maps compete on
marginal likelihood, which automatically penalises the extra parameters. At 40
labels it picks the plain linear model; past a couple of hundred it starts
preferring curvature, which is exactly right.

**Thompson sampling** over that exact posterior is the exploration policy. One
coherent hypothesis is drawn per slate and played out, rather than hedging
towards the safe middle.

### 4. Cold start

On day one the posterior is the prior, so the engine has to ask. Asking costs
your patience, the scarcest resource in the system, so the question is never
"which titles are good" but "which titles, once answered, most reduce
uncertainty about this person".

- **Phase one — diverse seed.** Greedy MAP inference over a determinantal
  point process. A DPP models *repulsion*, so it will not hand you six
  acclaimed American prestige dramas — once it takes one, the rest become
  redundant.
- **Phase two — adaptive questions.** For a Bayesian linear model the
  information gain from observing item `x` is exactly `½·log(1 + β·xᵀΣx)`, so
  the optimal next question is the title the model is least certain about.
  Classical sequential D-optimal design, closed form, no approximation.

Both phases only ask about titles you plausibly recognise, with the bar set
per-language.

### 5. Discovery — what it learned about you

`ent taste` inverts the representation. It takes the latent axes your
preference weights actually fire on, pulls the titles at each pole, and
describes what separates them using weighted log-odds with an informative
Dirichlet prior (Monroe, Colaresi & Quinn 2008) — the standard fix for naive
frequency comparison reporting whatever is common everywhere.

Nothing in the pipeline declares that an axis shall mean "pacing". PCA
produces whatever directions carry variance; your verdicts put weight on some
of them; which ones turn out to matter is an empirical fact about you.

---

## Does it work?

`ent eval` answers this without waiting months for your own history to
accumulate. MovieLens users **held out of collaborative-filtering training**
are replayed as cold-start strangers: the engine asks its questions, they
answer from their real rating history or say "haven't seen it", and after a
fixed answer budget the ranking is scored against their held-out ratings.

- **No leakage** — held-out users never touched the ALS fit; their
  elicitation and evaluation ratings are disjoint splits.
- **Same information to every arm** — baselines receive exactly the answers
  the full model received, so a win cannot come from asking better questions.
  Elicitation is measured separately by varying the budget.

Baselines: popularity, shrunk quality prior, liked-item centroid, rating-
weighted kNN, and plain ridge regression on identical features (the model
minus its Bayes). Reported with NDCG/precision/MAP/MRR **and** novelty,
diversity and serendipity, so an accuracy gain bought by collapsing onto the
canon is visible rather than hidden. Significance by paired bootstrap.

See [`docs/RESULTS.md`](docs/RESULTS.md).

---

## Install

Requires Python 3.11+ and roughly 40 GB of disk for the raw datasets. A CUDA
GPU makes the encoding pass minutes instead of hours; it is not required.

```bash
uv venv && uv pip install -e ".[encode,dev]"
cp .env.example .env     # then paste your free TMDB key
```

```bash
ent setup                # download, build, enrich, embed, factorise, fuse
```

`setup` is resumable — every stage skips work already done.

---

## Use

```bash
ent onboard --n 40                   # cold start, adaptively chosen questions
ent recs                             # what to watch
ent recs --lang ml,ta --movies -k 15
ent recs --series --since 2020
ent recs --strategy mean             # no exploration, safest picks
ent recs --explore 2.5               # feeling adventurous
ent recs --novelty 1.0               # push away from the canon
ent similar "Jallikattu"             # pure geometry, ignores your profile

ent loved "Jallikattu"               # teach it, by name
ent disliked "Morbius"
ent bulk my_films.txt                # one title per line, optional `| verdict`
ent add "Thaneer Mathan Dinangal"    # too obscure for the vote floor? fetch it anyway

ent taste                            # what it worked out about you
ent why "Memories of Murder"         # why it thinks you'd like this
ent forget "Morbius"                 # remove a verdict entirely
ent history
ent stats
ent audit                            # is it actually learning you?

ent eval --users 300 --budget 30     # benchmark against baselines
ent export                           # back up your verdicts
```

`◆` marks a confident pick, `◇` an exploratory one — the model is uncertain
and is spending a slot to find out. Both are logged with the probability they
had of being shown, which is what makes honest measurement of improvement
possible later.

---

## Design notes

**Why not a fine-tuned LLM or a generative recommender?** Semantic-ID methods
like TIGER and HSTU are genuinely the state of the art — for platforms with
millions of users and billions of interactions. Their advantage comes from
sequence modelling over enormous interaction logs. Here there is one user and
no reliable consumption timestamps, so that advantage does not exist, while
the costs — compute, opacity, and an inability to say how confident it is —
all do. Using them here would be cargo-culting scale.

**Why is the event log the source of truth?** The model is a pure function of
your verdicts, refitted from scratch on every command. It costs milliseconds
because the posterior is closed form, and it removes an entire category of bug
where a cached model silently drifts out of sync with the data.

**Why does quality get learned rather than blended in?** Most recommenders
hard-blend a popularity prior with a tuned coefficient, quietly deciding on
your behalf how much you should care what everyone else thinks. Here consensus
quality is just another input feature, and the posterior works out whether you
track critical consensus, ignore it, or run against it.

---

## Privacy

`.env`, `data/`, and every rating you give are gitignored. Nothing about your
taste is uploaded anywhere. TMDB is contacted only to fetch public catalogue
metadata, never to send anything about you.

## Licence

MIT.
