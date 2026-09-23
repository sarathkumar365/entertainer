# The story of how Entertainer learns your taste

## Before you start — the short version

If you have run full-stack web apps before, you already know most of this.

**There is no model server.** Nothing like Ollama or a ChatGPT-style process
sits in the background. There are two "models", and neither needs to stay
running:

1. **The embedding model (Qwen3-Embedding-0.6B).** It reads each film's plot
   and turns it into a list of numbers. It runs once, during the build, and
   the results are saved to disk. Afterwards it only wakes up briefly for
   extras such as adding a brand-new title or searching by mood.
2. **Your taste model.** This is a small piece of maths fitted to your
   ratings. It is recalculated from scratch, in milliseconds, every time you
   ask for recommendations.

So "starting the app" just means starting an ordinary web server — a Python
backend and a React frontend. The database is a single DuckDB file
(`data/entertainer.duckdb`), like SQLite, so there is no database server to
start either.

| Piece | Full-stack equivalent |
|---|---|
| `ent` | the project CLI, like `rails` or `artisan` |
| App, `http://127.0.0.1:8756` | the website: rate titles, get recommendations |
| Build Studio, `http://127.0.0.1:8757` | a read-only progress page, like a CI dashboard |
| `ent setup` (the build) | seeding and migrating a very large database — hours, but resumable |

Build Studio shows the eight stages, how far the current one has got (the
long TMDB and encoding stages report their own counts and a rough time
remaining), and — behind the "what is happening" toggle — the build's log of
what started, finished, was skipped or failed.

The build must finish before the app can run. While it is running, it holds
the database exclusively, so starting the app fails with a DuckDB lock error.
That is expected.

### What the eight build steps actually do

1. **Downloading source data.** Fetches IMDb's public film lists (every
   title, its rating, cast and crew) and MovieLens (32 million ratings from
   real people). Think of it as downloading a huge seed dump.
2. **Building the catalogue.** Merges those files into one table of films and
   shows, dropping titles almost nobody has voted on.
3. **Enriching from TMDB.** Asks TMDB, one title at a time, for the plot
   summary, keywords, poster and original language — things IMDb does not
   have. The plot matters most, because it is where the model learns what a
   film *feels* like. This is the slowest step: one network call per title,
   which can mean a few hours.
4. **Pruning by language.** Now that each title's real language is known,
   drops the ones that are too obscure *for their own industry*. The bar is
   fair per language: 200 votes is popular for a Malayalam film, but not for
   a Hollywood one.
5. **Encoding the text.** The AI model finally runs. It downloads once
   (about 1.2 GB), then reads every plot and writes each title as 256
   numbers. Films with a similar feel end up with similar numbers — mood,
   not genre. Slow without an NVIDIA GPU, so expect hours on a laptop.
6. **Factorising MovieLens.** Mines the 32 million ratings for "people who
   liked X also liked Y" patterns that no plot summary states. Plain maths,
   no AI model. Minutes.
7. **Fusing the item space.** Combines the "feel from the plot" numbers and
   the "people like me" numbers into one map. Every title gets a position;
   titles close together are ones you would probably feel the same about.
8. **Learning the population prior.** Learns what typical human taste looks
   like, so that with only a handful of your ratings the model makes
   sensible guesses rather than random ones.

When it prints `ready`, run `./scripts/entertainer start`. From then on,
your taste is simply your ratings measured against that map, recalculated
instantly on every request. Nothing heavy runs while you use the app.

The rest of this document tells the same story in depth.

Imagine the feature we want to build is simple to describe:

> “I tell the app what I liked. It should help me choose what to watch next, explain how confident it is, and eventually prove whether it is helping.”

That promise is one feature from the outside. Inside, it is a chain of smaller capabilities that must be built in order. A recommendation cannot exist until films can be compared. Films cannot be compared until their information is turned into a common format. A personal prediction cannot exist until your opinions are safely recorded. And none of it should be trusted until it is tested without letting the model rewrite history.

This guide follows that chain.

## Chapter 1 — First, we need a world to recommend from

Before machine learning enters the picture, the app must answer: “what films and shows do we know about?” A small list of popular English titles would be easy to build, but it would fail the product promise for Malayalam, Tamil, Telugu, Korean, Japanese, and other cinema. So our first job is to make a broad local catalogue.

~~~mermaid
flowchart LR
    imdb[IMDb] --> base[Base catalogue]
    tmdb[TMDB] --> enrich[Enriched film details]
    movieLens[MovieLens] --> audience[Audience connection data]
    base --> catalogue[Local catalogue]
    enrich --> catalogue
    audience --> catalogue
~~~

Each source exists because it fills a different gap:

- **IMDb** gives stable title IDs, years, credits, and broad international coverage. It answers “what exists?”
- **TMDB** gives synopses, posters, keywords, and original language. It answers “what is this film like?”
- **MovieLens-32M** gives millions of anonymous public ratings. It answers “which titles tend to be enjoyed by similar viewers?”
- **Your ratings** remain separate. They answer “what does this person like?”

A technical implication matters here. A highly regarded Malayalam film naturally has fewer global votes than a major English release. If we used one worldwide popularity cutoff, we would silently remove the former. Therefore, the pipeline learns each title’s original language from TMDB and applies language-aware quality thresholds.

At the end of this chapter we have a searchable catalogue. It still does not know that two films are similar, so we build the next capability: a common film map.

## Chapter 2 — Then, we turn every film into a place on a map

People can read two synopses and recognise that they feel related. A computer needs that relationship expressed as numbers. This numeric description is an **embedding**. The useful mental picture is a map: films with similar meaning should be close together.

~~~mermaid
flowchart TD
    card[Film card: title, synopsis, cast, language, keywords] --> encoder[Multilingual text encoder]
    encoder --> content[Content fingerprint]
    ratings[MovieLens rating patterns] --> collaborative[Collaborative fingerprint]
    content --> fused[Fused film map]
    collaborative --> fused
~~~

We need two kinds of evidence because either one alone has blind spots.

First, we create a compact text card for each title. A multilingual Qwen embedding model turns it into a **256-number content fingerprint**. This gives an obscure, recent, or regional title a meaningful position even if few people have rated it.

Text cannot capture every connection. Two films can have little in common on the page, yet fans of one often love the other. So we also factorise MovieLens’ user–title matrix with **implicit alternating least squares (iALS)**. That creates **192 collaborative factors** for titles MovieLens knows.

Most catalogue titles are not in MovieLens. Instead of pretending they have collaborative evidence, the system estimates the recoverable part from content and marks it lower-confidence. It then combines content and collaborative signals and uses PCA to create one **192-dimensional fused film map**.

This unlocks similar-title search. More importantly, it gives the personal model a shared language in which to understand your ratings.

## Chapter 3 — Your clicks become durable personal history

When you press **love**, **like**, **fine**, or **dislike**, the app does not quietly change a mysterious model file. It records an event in a local DuckDB database first. The event log is the source of truth; the model is a repeatable calculation made from that history.

~~~mermaid
flowchart LR
    click[Feedback click] --> event[Local rating event]
    event --> history[Complete viewing history]
    history --> fit[Refit current taste model]
    fit --> score[Scores and uncertainty]
~~~

This design is needed for three reasons:

1. Refreshing the page cannot erase a rating; it is in the local database.
2. Changing your mind creates a later event; the latest explicit verdict becomes the current label.
3. When we improve the model, we can rebuild it from the same history and compare fairly.

The feedback app can work before the full catalogue build finishes: it can save ratings, show a rated view, and keep rated titles out of the feedback feed. Full personal recommendations must wait for the fused film map from Chapter 2.

## Interlude — How you operate the system day to day

There are two different moments in the system's life, and separating them
prevents a lot of confusion.

The first is the **heavy build**. It constructs the shared film map: catalogue,
embeddings, MovieLens collaboration, fusion, and the optional starting prior.
It can take a long time, but it is resumable. The second is **using the app**.
Once the map exists, the app reads your local rating history and quickly refits
your small personal model whenever it needs a fresh recommendation. It does not
redo the expensive shared build every time you open the app.

~~~mermaid
flowchart TD
    build[One resumable model build] --> artifacts[Local catalogue and model artifacts]
    artifacts --> use[Start the local app]
    ratings[Your saved ratings] --> refit[Fast personal refit]
    artifacts --> refit
    refit --> slate[Recommendations and predictions]
    ratings --> next[Later ratings]
    next --> refit
~~~

The helper command gives each action a memorable name:

```bash
./scripts/entertainer build    # heavy resumable build + Build Studio
./scripts/entertainer start    # app and Build Studio
./scripts/entertainer status   # process state and available artifacts
./scripts/entertainer logs app # follow app, studio, or build logs
./scripts/entertainer debug    # readiness and recent app errors
./scripts/entertainer stop     # stop the managed local servers
./scripts/entertainer restart  # stop and start both, after a code change
./scripts/entertainer ui       # rebuild the browser interface
```

`build` is intentionally foreground work: you can see its terminal output and
stop it with Ctrl-C; running it again resumes completed stages. `start` is for
normal use after a build. The helper writes only PID files and logs under
`data/runtime/`, while ratings and model artifacts remain in their normal local
data locations. During a build, the local database may be write-locked; this is
intentional protection against the app and pipeline changing it at once.

There is a third moment, rarer than the other two: **changing the interface**.
The browser app is a built bundle committed to the repository, so running it
needs nothing but Python. Editing it needs Node, and `ui` is the only command
that does — it installs the dependencies once, rebuilds from
`src/entertainer/web/ui`, and writes content-hashed filenames so a reload
picks the new bundle up without any cache to clear.

### What the app shows you, and why it is split up

| screen | what it is for |
|---|---|
| `/` | the model itself — it grows as you feed it and its edge steadies as it gets surer |
| `/recs` | what to watch, each prediction drawn as the distribution it actually is |
| `/rate` | a grid for scanning, or a keyboard focus mode for volume |
| `/library` | saved, watched, rated |
| `/taste` | the learned axes, and your position among the catalogue |
| `/evidence` | whether it is working |

Two distinctions are load-bearing here, and both are easy to blur.

**Saving is not a verdict.** A save says you intend to watch something. It
teaches the model nothing, it is reversible, and its only effect is to keep
the title out of future slates. A verdict says what you thought: it trains the
model, and the only way to change one is to record another. They share the
same append-only log, which is why "saved" means *the most recent watchlist
event for this title says active* rather than *a watchlist event exists*.

**Where a verdict is given changes what it can prove.** Every slate records
the probability each title had of being shown. A verdict given on `/recs` can
therefore be matched back to the slate that produced it and weighted by that
probability, which is what makes "were these recommendations any good?" an
answerable question. The same verdict given on `/rate` teaches the model just
as much, but proves nothing about the recommender. That is why `/evidence`
reports the off-policy estimate separately, and hedges it.

## Chapter 4 — The model turns history into a taste estimate

We now have the two ingredients needed for personal prediction: your verdicts and a numeric fingerprint for every film. The next question is: “which directions on the film map explain what you tend to enjoy?”

~~~mermaid
flowchart LR
    ratings[Saved ratings] --> model[Bayesian taste model]
    map[Fused film map] --> model
    model --> score[Predicted score: 0 to 10]
    model --> likelihood[Chance you will like it]
    model --> uncertainty[Uncertainty range]
    model --> ranking[Recommendation ranking]
~~~

The core model is **Bayesian ridge regression**.

- **Regression** means it learns a numeric preference score.
- **Ridge** means it is restrained: a few ratings should not make it invent a dramatic story about you.
- **Bayesian** means it keeps uncertainty, not only one confident answer.

Technically, each rated title supplies a feature vector x and a reward y, where the verdict is mapped to a 0–1 scale. The model learns weights w:

    preference approximately equals x dot w plus noise

The posterior mean of w produces an expected score. Its covariance produces the uncertainty range. The fit is closed-form, so it can be rebuilt from your small history in milliseconds.

A few protections make that fit behave sensibly:

- Recent ratings count a little more, with a 1,100-day half-life. Old opinions still matter, but cannot freeze your taste forever.
- Skips are weak negative evidence: “not tonight” is not “bad.”
- The model adds 1,000 weak, deterministic samples from unseen titles. This gives contrast to a history full of films you chose to watch, without pretending unseen titles are explicit dislikes.
- A MovieLens-derived population prior makes the first few ratings less random. As your own evidence grows, its influence fades.
- A random-Fourier-feature lift can add gentle non-linearity, but only when marginal likelihood says the real number of ratings supports extra complexity.

That final guardrail is important. The project does not assume the more complex model is better; plain ridge remains a serious competing model.

## Chapter 5 — A taste estimate becomes a useful top ten

A high predicted score alone does not make a good slate. The system must obey your constraints, avoid repeats, avoid ten near-identical films, and sometimes learn something new.

~~~mermaid
flowchart TD
    map[Fused film map] --> candidates[Unseen candidate titles]
    filters[Language, format, year, runtime filters] --> candidates
    candidates --> scoring[Score with taste posterior]
    scoring --> shortlist[Keep strongest shortlist]
    shortlist --> diversify[Diversify with MMR]
    diversify --> slate[Top-ten recommendation slate]
    slate --> log[Log score, uncertainty, policy, propensity]
~~~

The ranker removes titles you already rated or dismissed. It applies hard filters—such as Malayalam only, movies only, or under two hours—before scoring. A hard requirement should never become a vague preference.

It then scores candidates using either the posterior mean (the safe ranking) or **Thompson sampling**. Thompson sampling draws one plausible version of your taste from the uncertainty distribution. This creates measured exploration: a surprising title can appear because it may fit and because your future verdict would teach the system something.

Finally, maximal marginal relevance (MMR) trades a little score for variety. That prevents a top ten made of near-duplicates. Every displayed title logs its score, uncertainty, selection policy, and propensity—the chance it had of being shown—so later results can be interpreted honestly.

## Chapter 6 — We do not call it good until it survives a fair test

At this point the system can make recommendations. That is not proof that it is useful. We need two tests because they answer different questions.

The **offline MovieLens replay** asks whether the engineering method works in a controlled public-data simulation. Some MovieLens users are held out of collaborative training, treated as new users, and compared with simple baselines. It protects against leakage, but cannot prove the model knows *your* taste.

The **sealed hidden-pool validation** asks the personal question. You add films you have watched but have not rated. Before seeing your answer, the app saves the full model’s and ridge model’s predictions and rankings. Then you reveal the verdict.

~~~mermaid
flowchart TD
    pool[Choose watched, unrated titles] --> seal[Seal full-model and ridge predictions]
    seal --> reveal[Reveal verdicts one by one]
    reveal --> compare[Compare saved predictions with reality]
    compare --> report[Write interval-aware evidence report]
~~~

Because the prediction is sealed first, it cannot improve after seeing the answer. “Not seen” stays unresolved and never affects the result.

The main product metric is **Top-10 hit rate**: among the films a model ranked highest, how often did you actually like them? The report also includes ranking quality (NDCG@10 and precision@10), score error (MAE/RMSE), calibration, and Brier score. Every personal result includes a 95% confidence interval.

Thirty completed hidden-pool outcomes are called preliminary, never trusted. At 100 outcomes, the full model stays only if its paired confidence interval proves a positive Top-10 lift over ridge. If it cannot, the complex population-prior and RFF path should be archived and ridge should serve the same interface instead. That is the system choosing evidence over complexity.

## The entire build story, condensed

~~~mermaid
flowchart TD
    start[We want reliable personal recommendations] --> world[Build a broad, fair film catalogue]
    world --> map[Turn films into one comparable map]
    map --> history[Save your ratings as durable local events]
    history --> taste[Fit a small uncertainty-aware taste model]
    taste --> slate[Rank, diversify, and log recommendations]
    slate --> test[Seal predictions and measure outcomes]
    test --> decision[Keep the full model only if it earns the result]
~~~

## What lives where

Your ratings, catalogue, model artifacts, and evidence reports live under the local data directory. The main database is data/entertainer.duckdb. Build and evaluation manifests record source hashes, artifact checksums, model settings, random seeds, split IDs, and timestamps, so a future result can be traced back to the exact build that produced it.

TMDB credentials live in a local .env file. The app uses TMDB only to fetch public metadata; it does not send ratings to TMDB, IMDb, MovieLens, or a central service. To move your history to another computer, use ent export and ent import privately.

## What to expect as a user

Early recommendations are informed guesses. A varied mix of **love**, **like**, **fine**, and **dislike** ratings gives the clearest signal. After the full build, the app can score and rank unseen titles; after sealed validation, you can see whether those rankings genuinely beat the simpler baseline.

That is the complete feature: not merely “a model made a list,” but a local system that learns, remembers, recommends, and shows its evidence.
