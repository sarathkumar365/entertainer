# entertainer

A personal recommendation engine for film and television that learns what you
like from **titles and verdicts alone** — no genres, no tags, no explaining
yourself — across English, Malayalam, Tamil, Telugu, Kannada, Korean,
Japanese, European and other catalogues.

> **New here?** Read [How Entertainer learns your taste](docs/HOW_IT_WORKS.md)
> for a plain-English tour of the data, model, predictions, and evaluation
> process. It is the best place to start before the technical details below.

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
same fidelity as Hollywood.

**TMDB** enriches each title with a synopsis, a curated keyword vocabulary,
and an authoritative `original_language`. The synopsis matters most: it is
the only place where pace, tone and moral posture are written down at all.

**MovieLens-32M** contributes 32 million ratings of co-consumption behaviour.

A vote threshold tuned for American films would erase most of the Malayalam
catalogue, so the floor is **per-language**: 200 votes on a Malayalam film is
a comparable cultural footprint to 2,000 on an English one.

Applying that floor requires knowing the language, and getting this wrong was
the most instructive failure in the project. IMDb's `akas` table has a
`language` column, which looks like exactly the right field and is not: it
records the language of a *localised release*, not of the film. Kantara, a
Kannada film, carries tags for English, French, Hindi, Japanese and Turkish
and none for Kannada. Baahubali carries Tamil, Telugu, Hindi and English with
nothing marking which is the original.

The resulting failure was silent and biased in one direction — every
non-English film with a US or UK release picked up an `en` tag, was then
measured against the English vote floor, and vanished. The catalogue came out
79% English with zero Tamil, Malayalam or Telugu titles, and looked entirely
plausible.

So the build now applies a flat floor and **declines to guess** a language.
Pruning happens afterwards, once TMDB has supplied one it can be trusted:
272,403 candidates in, language-aware floors applied after enrichment.

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

### 3b. The prior knows what taste looks like

An isotropic prior asserts something obviously false: that on day one, before
any evidence, every direction in taste space is equally plausible. Real
preference vectors live on a thin, structured manifold — nobody's taste is a
random direction in 192 dimensions.

MovieLens holds two hundred thousand examples of what a real preference vector
looks like. Fitting one weight vector per user *in the same fused space*, then
taking the mean and covariance of that population, gives a prior that already
knows the shape of human taste before its own user has answered anything. This
is empirical Bayes at the population level, and it stays fully closed form: a
block-diagonal reparameterisation folds the population covariance into the
isotropic problem already being solved.

The payoff sits exactly where it is needed. Past forty answers the likelihood
dominates and the prior barely registers. At five, the prior is most of the
posterior, and the difference between "any direction is equally likely" and
"directions look like this" is the difference between a useful first slate and
a random one.

One subtlety cost some measurement to find. Estimating the prior precision by
maximum likelihood — correct, and what the model does everywhere else —
collapses at small n in the badly-conditioned whitened basis and produces a
confidently wrong direction. Anchoring it with a weak hyperprior turns a 0.39
correlation *loss* at n=4 into a 0.46 gain, and still decays to nothing by
n=64. The benchmark runs the engine with and without the population prior as
separate arms, so the gain is measured rather than claimed.

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
weighted kNN, plain ridge regression on identical features (the model minus
its Bayes), and the full engine with an isotropic prior (the model minus its
population knowledge). Reported with NDCG/precision/MAP/MRR **and** novelty,
diversity and serendipity, so an accuracy gain bought by collapsing onto the
canon is visible rather than hidden. Significance by paired bootstrap.

The protocol is fixed in advance and written down in
[`docs/RESULTS.md`](docs/RESULTS.md), along with the results — deciding what
counts as success after seeing the numbers is how offline recommender
evaluation usually goes wrong.

**Current standing, stated plainly: the engine has not earned its
complexity.** On 300 held-out users it is statistically indistinguishable
from plain ridge regression on the same features (Δ=−0.0119, p=0.96), from a
weighted kNN, and from ranking by vote count. It reaches that accuracy at
higher novelty and non-zero serendipity, which is a real difference and also
not what NDCG measures. The one component whose value is beyond argument is
implicit negatives, worth +0.157 (p<0.0001).

Two earlier runs looked better and were wrong — one measured a degenerate
fit, the other measured handicapped baselines. Both are kept in the results
document, because the flattering run is the one that would otherwise have
been quoted.

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

`setup` is resumable — every stage skips work already done. The catalogue
and the MovieLens factorisation are skipped outright when their inputs are
unchanged since they last succeeded (`--force` rebuilds them anyway), TMDB
only fetches titles it has never seen, and the encoder only encodes cards
whose text changed. The factorisation runs in a background process
alongside the TMDB and encoding stages; `--no-parallel` runs it in line.

Memory is planned as a share of physical RAM — half by default, set with
`--memory-fraction 0.3` or `ENTERTAINER_MEMORY_FRACTION`. DuckDB is capped
at that share; the background factorisation only starts when it and the
encoder fit inside it together, so an 8 GB laptop runs the stages one after
another while a larger server overlaps them.

On a Linux machine with an NVIDIA GPU, `uv pip install -e ".[encode,web,gpu]"`
takes torch from PyTorch's CUDA 12.8 index (driver 570 or newer). The encoder
then runs in bf16 with a batch size sized to free VRAM; `ENTERTAINER_DEVICE`
overrides the device choice and `ENTERTAINER_CF_GPU=0` keeps the
factorisation on the CPU.

---

## Everyday controls

You do not need to remember ports, background-process commands, or log paths.
From the project folder:

```bash
./scripts/entertainer start    # the app and Build Studio
./scripts/entertainer status   # what is running, and which artifacts exist
./scripts/entertainer logs app # follow the app log (or: studio, build)
./scripts/entertainer debug    # readiness plus recent app errors
./scripts/entertainer stop     # stop the managed local services
./scripts/entertainer restart  # stop and start both, after a code change
./scripts/entertainer ui       # rebuild the browser interface after editing it
```

`start` is for using an already-built recommender. It puts the app on
`http://127.0.0.1:8756` and Build Studio on `http://127.0.0.1:8757`. `stop`
only stops processes this helper launched; it never deletes ratings,
catalogue data, model artifacts, or reports.

The equivalent without the helper is `ent rate`, which runs in the
foreground and opens a browser.

### The interface

Six screens, each showing one thing rather than all of them at once:

| | |
|---|---|
| `/` | the model itself, drawn as a living thing — it grows with what you tell it and tightens as it gets surer |
| `/recs` | what to watch, with every prediction drawn as a distribution instead of a number |
| `/rate` | a poster grid for scanning, or a keyboard-driven focus mode for getting through a lot |
| `/library` | what you have saved, watched, and rated |
| `/taste` | the axes it learned, and where you sit among 73,000 titles |
| `/evidence` | whether any of it is working |

Saving something is deliberately **not** a verdict. It says you intend to
watch it, never that you liked it, so it teaches the model nothing — it only
stops the title being recommended again. A verdict is the opposite: it trains
the model, and the only way to change one is to record another.

Rating a title **from `/recs`** is worth more than rating the same title
anywhere else. Recommendations are logged with the probability each had of
being shown, so a verdict given there is the one thing that can answer "were
these slates actually good?". Nowhere else produces that measurement.

### Changing the interface

The bundle is committed, so running the app needs no `node`. Editing it does:

```bash
./scripts/entertainer ui       # installs deps on first run, then builds
```

The source is `src/entertainer/web/ui`. Bundle filenames are content-hashed,
so a rebuild arrives without clearing any cache.

For a fresh or incomplete model build, use:

```bash
./scripts/entertainer build
```

It starts Build Studio at `http://127.0.0.1:8757`, runs the resumable build
in the foreground, and saves its terminal output under `data/runtime/logs/`.
The recommendation app is at `http://127.0.0.1:8756` after `start`.

While a build is actively writing the catalogue, `status` may say that model
details are temporarily unavailable. That is normal: DuckDB gives the build
exclusive write access so it cannot race the app. Build Studio remains the
right place to watch progress; when the build finishes, `status` will show
the finished artifacts.

Opens a local page of posters: recent titles that were well received *in
their own industry*, balanced across languages, with five buttons each —
loved, liked, fine, disliked, haven't seen. Click through them.

The engine learns from verdicts and nothing else, so how fast you can give
verdicts is the rate-limiting step in making it good. Recognising a poster is
much quicker than recalling and typing a transliterated title, and the
friction compounds over the hundred-odd ratings the model actually needs.

Language mix is adjustable by clicking the chips; the defaults lean towards
Malayalam, Tamil and English and keep Telugu deliberately quiet. The search
box finds anything in the catalogue *and* anything TMDB knows that is not —
searching `പ്രേമലു` returns Premalu even though the catalogue has no
Malayalam-script index. Rating a title from that second group pulls it in,
encodes it and places it in the item space, with no rebuild.

Nothing leaves the machine but TMDB metadata requests.

### On a second machine

A clone with no catalogue serves the same interface straight from TMDB — no
data to move, just a TMDB key in `.env`:

```bash
ent rate            # live mode is chosen automatically when there is no catalogue
```

The trade is real: TMDB's rating pool for South Indian cinema is roughly two
orders of magnitude thinner than IMDb's, so the live feed is shallower and
skews towards whatever had international distribution. For the full
catalogue, move a bundle across:

```bash
ent bundle export --no-space --path rate.zip     # 21MB, enough to rate and search
ent bundle export --path full.zip                # 75MB, adds recommendations
```

Verdicts collected anywhere merge back by IMDb id:

```bash
ent export --path verdicts.jsonl     # on the second machine
ent import verdicts.jsonl            # on the main one
```

## Use

```bash
ent rate                             # the browser interface — the fast way to teach it
ent onboard --n 40                   # or cold start in the terminal
ent recs                             # what to watch
ent recs --lang ml,ta --movies -k 15
ent recs --series --since 2020
ent recs --strategy mean             # no exploration, safest picks
ent recs --explore 2.5               # feeling adventurous
ent recs --novelty 1.0               # push away from the canon
ent recs --mood "slow, quiet, ambiguous ending, no action"
ent similar "Jallikattu"             # pure geometry, ignores your profile

ent loved 3                          # teach it, by slate position
ent loved "Jallikattu"               # or by name
ent disliked "Morbius"
ent dismiss "Emily in Paris"         # not interested, never watched it
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

**Why is the mood text a nudge rather than a filter?** Hard constraints —
language, film versus series, runtime — are applied as filters, never as
score penalties, because a constraint expressed as a penalty produces a slate
that *mostly* obeys it, which is worse than useless when the constraint was
the point. A mood is the opposite: "something slow tonight" is a preference,
not a requirement, so it is blended on a standardised scale where weight 1.0
means a title one standard deviation more on-mood outranks one a standard
deviation more to your taste. The query text is embedded and projected into
the same fused space through the stored imputation map, so it is compared
against titles in exactly the geometry the model reasons in.

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

`.env`, `data/`, `profile*.jsonl` and every rating you give are gitignored.
Nothing about your taste is uploaded anywhere. TMDB is contacted only to fetch
public catalogue metadata, never to send anything about you.

`tests/test_packaging.py` asserts those exclusions still hold, and that no
source file is ignored — the two failure modes are opposites and a single
careless pattern causes both. A bare `data/` once matched
`src/entertainer/data/` and silently excluded the whole ingest layer from the
repository, while `profile.jsonl` — the default export of your entire viewing
history — was not excluded at all.

## Licence

MIT.
