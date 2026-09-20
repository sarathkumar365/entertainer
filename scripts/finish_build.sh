#!/usr/bin/env bash
# Run the remainder of the pipeline once TMDB enrichment finishes.
#
# Ordering is load-bearing:
#   keywords before the rebuild, so the rebuild carries them across;
#   rebuild before prune, so MovieLens identities and cast credits are present;
#   prune before embed, so nothing is encoded that is about to be deleted;
#   embed before fuse, and fuse before the population prior, which lives in
#   the fused space.
set -euo pipefail

ENT="${ENT:-.venv/bin/entertainer}"
LOG_DIR="${LOG_DIR:-/tmp/entertainer-build}"
ENRICH_PID="${ENRICH_PID:-}"
mkdir -p "$LOG_DIR"

step() { echo; echo "=== $* ($(date +%H:%M:%S)) ==="; }

if [[ -n "$ENRICH_PID" ]]; then
  step "waiting for TMDB enrichment (pid $ENRICH_PID)"
  while kill -0 "$ENRICH_PID" 2>/dev/null; do sleep 60; done
fi

step "keyword backfill"
$ENT data keywords --top 150000 2>&1 | tee "$LOG_DIR/keywords.log" | tail -3

step "waiting for title.principals, if still downloading"
for _ in $(seq 1 120); do
  if [[ -f data/raw/imdb/title.principals.tsv.gz ]]; then
    a=$(stat -c%s data/raw/imdb/title.principals.tsv.gz)
    sleep 45
    b=$(stat -c%s data/raw/imdb/title.principals.tsv.gz)
    [[ "$a" == "$b" ]] && break
  else
    sleep 45
  fi
done

step "rebuild (picks up MovieLens identities and cast; carries enrichment)"
$ENT data build --min-votes 50 2>&1 | tee "$LOG_DIR/build.log" | tail -6

step "prune by language"
$ENT data prune 2>&1 | tee "$LOG_DIR/prune.log" | tail -30

step "encode item cards"
$ENT data embed --batch-size 64 2>&1 | tee "$LOG_DIR/embed.log" | tail -4

step "fuse item space"
$ENT data fuse 2>&1 | tee "$LOG_DIR/fuse.log" | tail -4

step "population prior"
$ENT data prior 2>&1 | tee "$LOG_DIR/prior.log" | tail -4

step "done"
$ENT stats 2>&1 | tee "$LOG_DIR/stats.log" | tail -40
