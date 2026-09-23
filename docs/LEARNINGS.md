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
- A handicapped baseline manufactures a win: giving ridge the sampled negatives and fixing kNN's normalisation erased an apparent significant victory entirely — fix the baselines before believing the result, not after.
- Keep the losing runs. The flattering one is the one that gets quoted, so the record has to contain the corrections beside it.
- A leave-one-out sweep on 169 labels ranked the shipped negative-sample count last and a 10x smaller one best; the held-out benchmark reversed it, so the sweep was noise. Tune on held-out users, never on the label set the sweep scores against.
- Benchmark arms move together between runs when the evaluated user sample changes: compare within a run against a baseline arm, never across runs on the absolute metric.
- Do not report a performance regression from a timing taken under load. A 582s test suite was 75s; load average was 23.7 on 12 cores because a benchmark, an ablation and a web app were running alongside it.

## External rating imports

- Netflix's exported `movieID` maps to nothing public, so a thumbs history can only be joined on title text, and title text is ambiguous: `Hunger`, `Alpha`, `Youth`, `Sahara` and `Extinction` each name several unrelated films.
- TMDB popularity is recency-weighted, so a 2026 release with 29 votes outranks the 1989 film actually watched — rank exact-title collisions on vote count, or better, refuse to rank them at all.
- `watch/providers` looked like decisive evidence that a candidate was the Netflix one, and is not: it reports *current* availability, and both test cases had already left Netflix. Verify a disambiguator before building on it.
- Grading matches (`high` / `ambiguous` / `low`) and writing only the unambiguous ones caught the two the heuristic would have got wrong; the user confirmed one of them unprompted.
- Two source rows can resolve to one film: Netflix lists the Chinese and Korean `A Love So Beautiful` separately with opposite verdicts, and writing both would put contradictory labels on a single catalogue row. Detect the collision rather than letting last-write win.
- Imported verdicts must carry their original timestamp. `now()` scrambles the prequential replay, which walks the log in `ts` order, and hands a two-year-old opinion full recency weight.
- Record human rulings in a file keyed by source id, not in a throwaway script: the next page of the same export then re-imports identically.
- Ask. Thirty-nine ambiguous titles were settled by the user in four rounds of questions, faster and more accurately than any amount of heuristic tuning would have managed.

## Process

- Session teardown kills the whole process group, `setsid` included in some cases — long jobs need a stable log path outside session-scoped directories, and resumability rather than trust.
- Any stage over ~10 minutes with nothing to show until the end needs checkpointing: the 40-minute encode was one teardown away from being lost entirely.
- Rich progress bars render nothing to a non-tty, so a background job needs explicit periodic logging or it looks hung.
