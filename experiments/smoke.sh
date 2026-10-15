#!/usr/bin/env bash
# Smoke: two tickets, every arm, one pass. Paid (about 2 tickets × 9 arms × ≤5 searches).
# Run from the repository root with .env filled and datasets/ built.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${ARMS:=perplexity_low,exa_fast,parallel_fast,jevlin_fast,jevlin_balanced,jevlin_single,jevlin_fanout_raw,jevlin_fanout_merged,jevlin_cli}"
: "${IDS:=hard_sap_business_partner_relationships,hard_sn_activate_shared_snapshot}"
: "${OUT_DIR:=runs/smoke}"
mkdir -p "$OUT_DIR"
PY="${PYTHON:-}"
[ -z "$PY" ] && { [ -x .venv/bin/python ] && PY=.venv/bin/python || PY=python3; }

# No --dataset: that flag names a Braintrust project and needs BRAINTRUST_API_KEY; tracing stays off.
DATASETS_ROOT=datasets PYTHONPATH=scripts "$PY" -m coding_search run \
  --split search-only \
  --backend "$ARMS" --ids "$IDS" \
  ${MODEL:+--model "$MODEL"} \
  --shuffle-seed 7 --concurrency 1 \
  --raw-dir "$OUT_DIR/raw" --out "$OUT_DIR/pass-1.json"

"$PY" scripts/jevlin_bench/analyze.py --out "$OUT_DIR/pass-1.json" \
  --catalog datasets/web-search/coding/tasks/catalog.json \
  ${RATE_CARD:+--rate-card "$RATE_CARD"} --md "$OUT_DIR/report.md"
