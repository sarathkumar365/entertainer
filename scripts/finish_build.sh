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

usage() {
  cat <<'EOF'
Usage: ./scripts/finish_build.sh
       ./scripts/finish_build.sh --help

Runs the rest of the pipeline after TMDB enrichment, in an order that is
load-bearing: keywords, rebuild, prune, embed, cf, fuse, prior, stats.
It takes no arguments; environment variables steer it:

  ENT=<path>         The CLI to call. Default: .venv/bin/entertainer
  LOG_DIR=<path>     Where each step's full log goes. Default: /tmp/entertainer-build
  ENRICH_PID=<pid>   Wait for this process to exit before starting.

Prefer `./scripts/entertainer build` for a normal full build; this script is
for resuming a run whose enrichment is already under way.
EOF
}

case "${1:-}" in
  -h|--help|help) usage; exit 0;;
  "") ;;
  *) echo "Unknown argument: $1" >&2; echo >&2; usage >&2; exit 2;;
esac

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

step "factorise MovieLens"
$ENT data cf --factors 192 --iterations 20 --holdout 2000 --signal watched 2>&1 | tee "$LOG_DIR/cf.log" | tail -4

step "fuse item space"
$ENT data fuse 2>&1 | tee "$LOG_DIR/fuse.log" | tail -4

step "population prior"
$ENT data prior 2>&1 | tee "$LOG_DIR/prior.log" | tail -4

step "done"
$ENT stats 2>&1 | tee "$LOG_DIR/stats.log" | tail -40
