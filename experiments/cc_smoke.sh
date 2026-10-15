#!/usr/bin/env bash
# Claude Code agent, smoke: a few tickets, every arm, one pass. Bills the Claude subscription, not the API.
# Run from the repository root with .env filled (vendor + Jevlin keys for the wrapper) and datasets/ built.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-}"
[ -z "$PY" ] && { [ -x .venv/bin/python ] && PY=.venv/bin/python || PY=python3; }

: "${ARMS:=perplexity_low,exa_fast,parallel_fast,jevlin_fast,jevlin_balanced,jevlin_single,jevlin_fanout_raw,jevlin_fanout_merged,jevlin_cli}"
: "${IDS:=hard_sn_activate_shared_snapshot,hard_wd_create_mentorship_for_employee}"
: "${MODEL:=claude-sonnet-5-5}"
: "${OUT_DIR:=runs/cc-smoke}"
mkdir -p "$OUT_DIR"

"$PY" experiments/catalog_freeze.py verify

DATASETS_ROOT=datasets PYTHONPATH=scripts "$PY" scripts/jevlin_bench/cc_run.py \
  --arms "$ARMS" --ids "$IDS" --model "$MODEL" --seed 7 \
  --out "$OUT_DIR/pass-1.json" --raw-dir "$OUT_DIR/raw" --jobs-dir "$OUT_DIR/jobs"

"$PY" scripts/jevlin_bench/analyze.py --out "$OUT_DIR/pass-1.json" \
  --catalog datasets/web-search/coding/tasks/catalog.json \
  --rate-card experiments/rates.json --md "$OUT_DIR/report.md"
