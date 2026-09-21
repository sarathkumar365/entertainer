# Failures and learnings

One line each, newest last. Written so the same mistake is not made twice.
Every entry is something that actually happened in this project, not a
general principle.

## Data

- IMDb `akas.language` is the language of a *localised release*, not the film: Kantara (Kannada) carries en/fr/hi/ja/tr and no kn — never infer production language from it, use TMDB `original_language`.
- That failure was silent and one-directional: every non-English film with a US release got tagged `en`, met the English vote floor, and vanished — catalogue came out 79% English with zero Tamil/Malayalam/Telugu and looked entirely plausible.
- Do not filter on a derived field before the field is trustworthy: apply a flat floor at build time and the language-aware floor after enrichment.
- An empty API result is a result: `keywords = []` must be distinguishable from "never fetched", or the backfill re-requests the same titles forever — use a `*_at` timestamp marker.
- A coverage flag (`--top N` = "ensure the N most-voted have keywords") is idempotent; a batch size ("fetch N more") silently moves on to the next N after an interruption.
- Enrichment writing `[]` over fetched keywords silently undid the backfill; a merge that can erase data needs a `CASE WHEN len(?) > 0` guard.

## Persistence

- Item ids are positional within a build, so a rebuild renumbers everything and the event log must be remapped through a stable key (IMDb id) — on the first real rebuild this moved 179,631 of 272,403 titles.
- A vote floor decides what to *offer*, not what to erase: never prune a title the user has rated, or their verdict is orphaned.
- A threshold applied to an unmeasured quantity is meaningless, not strict: hand-added titles have no IMDb votes and must be exempt rather than treated as zero.
- Positional `INSERT INTO t SELECT ...` breaks on the next migration, and breaks silently if the column counts happen to match — always name the columns.
- DuckDB is columnar: 272,403 single-row `UPDATE`s took 4+ minutes and climbing; one set-based `UPDATE ... FROM` temp table took 16.7 seconds.

## Repository hygiene

- A gitignore pattern without a leading slash matches at any depth: a bare `data/` excluded `src/entertainer/data/` and kept the entire 1,100-line ingest layer out of the repo for twenty commits, with a fresh clone unable to run.
- ruff respects `.gitignore` too, so the same pattern meant that code was never linted either.
- The inverse of that bug shipped at the same time: `profile.jsonl`, the default `ent export` target holding the user's whole viewing history, was *not* ignored in a public repo.
- Tests that run against the working tree cannot notice either problem — assert explicitly that no source file is ignored and that every secret path is.

## Modelling

- Verify the claim you write in the comment: the "D-optimal chases unrepresentative outliers" rationale failed its own test on the fixtures and had to be removed.
- A synthetic fixture that saturates proves nothing — everything hit r≈0.97 past twenty questions, so the elicitation comparison had to move to the MovieLens replay.
- Unconstrained diversity is not balanced diversity: a DPP over a catalogue that is 46% English opens with all-English questions, and trimming afterwards does not help because the other languages were never selected — stratify the search itself.
- Empirical Bayes on the prior precision collapses at small n in a badly-conditioned whitened basis; anchoring alpha with a weak hyperprior turned a 0.39 correlation loss at n=4 into a 0.46 gain.
- iALS spreads variance evenly across its dimensions but only the leading ones are predictable from text: keeping all 192 gave imputation R^2 0.074, truncating to 48 gave 0.138 for 1.8% of the similarity structure.
- **Centering a compressed reward distribution destroys the signal**: MovieLens verdicts were 0.789 ± 0.092, so centering turned "liked slightly less" into "disliked" — the Bayesian fit shrank every weight to zero (alpha hit its 1e6 clip) and predicted a constant, and the same bug sank weighted-kNN.
- A preference model trained only on things the user rated has never seen "not for me": adding sampled unrated negatives took NDCG@10 from 0.0922 to 0.1970.
- A catastrophic-looking result can be an interaction, not a cause — the population prior scored 0.0036 only because the underlying fit was degenerate; with negatives it is neutral.
- Always run the real benchmark before believing a component helps: three separate things that improved on synthetic data did nothing or harmed on held-out users.

## Process

- Session teardown kills the whole process group, `setsid` included in some cases — long jobs need a stable log path outside session-scoped directories, and resumability rather than trust.
- Any stage over ~10 minutes with nothing to show until the end needs checkpointing: the 40-minute encode was one teardown away from being lost entirely.
- Rich progress bars render nothing to a non-tty, so a background job needs explicit periodic logging or it looks hung.
