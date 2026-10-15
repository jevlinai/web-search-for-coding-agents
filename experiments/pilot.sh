#!/usr/bin/env bash
# Pilot: every scorable public ticket, every arm, PASSES passes with distinct shuffle seeds.
# Paid. Run from the repository root with .env filled and datasets/ built.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${ARMS:=perplexity_low,exa_fast,parallel_fast,jevlin_fast,jevlin_balanced,jevlin_single,jevlin_fanout_raw,jevlin_fanout_merged,jevlin_cli}"
: "${PASSES:=3}"
: "${MAX_RESULTS:=8}"
: "${OUT_DIR:=runs/pilot}"
mkdir -p "$OUT_DIR"
PY="${PYTHON:-}"
[ -z "$PY" ] && { [ -x .venv/bin/python ] && PY=.venv/bin/python || PY=python3; }

# The ticket tree must be the frozen one; a changed catalog would be a different experiment.
"$PY" experiments/catalog_freeze.py verify

for pass in $(seq 1 "$PASSES"); do
  OUT="$OUT_DIR/pass-$pass.json"
  RESUME=()
  if [ -f "$OUT" ]; then
    # An interrupted pass: keep its rows without a run error, run only the rest (same seed order).
    cp "$OUT" "$OUT_DIR/pass-$pass.before-resume-$(date -u +%Y%m%dT%H%M%SZ).json"
    RESUME=(--resume-from "$OUT")
  fi
  echo "== pass $pass / $PASSES  arms=$ARMS  max_results=$MAX_RESULTS ${RESUME:+(resuming)}"
  # No --dataset: that flag names a Braintrust project and needs BRAINTRUST_API_KEY; tracing stays off.
  DATASETS_ROOT=datasets PYTHONPATH=scripts "$PY" -m coding_search run \
    --split search-only \
    --backend "$ARMS" \
    ${MODEL:+--model "$MODEL"} \
    --max-results "$MAX_RESULTS" \
    --shuffle-seed "$pass" --concurrency 1 \
    --abort-after-run-errors 5 "${RESUME[@]}" \
    --raw-dir "$OUT_DIR/raw" --out "$OUT"
  # A pass that aborted (model outage, no credits) stops the whole run; fix the cause and rerun.
  if "$PY" -c "import json,sys; sys.exit(0 if json.load(open('$OUT')).get('aborted') else 1)"; then
    echo "pass $pass aborted; stopping. Rerun pilot.sh to resume."
    exit 1
  fi
done

"$PY" scripts/jevlin_bench/analyze.py --out "$OUT_DIR"/pass-*.json \
  --catalog datasets/web-search/coding/tasks/catalog.json \
  ${RATE_CARD:+--rate-card "$RATE_CARD"} --md "$OUT_DIR/report.md" --json "$OUT_DIR/report.json"
