#!/usr/bin/env bash
# Claude Code agent, pilot: every frozen ticket, every arm, PASSES passes with distinct seeds.
# Bills the Claude subscription; a spent usage window is waited out and retried by the driver.
# Rerunning resumes an interrupted pass (rows without a run error are kept).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-}"
[ -z "$PY" ] && { [ -x .venv/bin/python ] && PY=.venv/bin/python || PY=python3; }

: "${ARMS:=perplexity_low,exa_fast,parallel_fast,jevlin_fast,jevlin_balanced,jevlin_single,jevlin_fanout_raw,jevlin_fanout_merged,jevlin_cli}"
: "${MODEL:=claude-sonnet-5-5}"
: "${PASSES:=3}"
: "${MAX_RESULTS:=8}"
: "${OUT_DIR:=runs/cc-pilot}"
: "${RATE_LIMIT_WAIT:=1200}"
mkdir -p "$OUT_DIR"

"$PY" experiments/catalog_freeze.py verify

for pass in $(seq 1 "$PASSES"); do
  OUT="$OUT_DIR/pass-$pass.json"
  RESUME=()
  if [ -f "$OUT" ]; then
    cp "$OUT" "$OUT_DIR/pass-$pass.before-resume-$(date -u +%Y%m%dT%H%M%SZ).json"
    RESUME=(--resume-from "$OUT")
  fi
  echo "== pass $pass / $PASSES  model=$MODEL arms=$ARMS max_results=$MAX_RESULTS ${RESUME:+(resuming)}"
  if ! DATASETS_ROOT=datasets PYTHONPATH=scripts "$PY" scripts/jevlin_bench/cc_run.py \
      --arms "$ARMS" --model "$MODEL" --seed "$pass" --max-results "$MAX_RESULTS" \
      --rate-limit-wait "$RATE_LIMIT_WAIT" --abort-after-run-errors 5 "${RESUME[@]}" \
      --out "$OUT" --raw-dir "$OUT_DIR/raw" --jobs-dir "$OUT_DIR/jobs"; then
    echo "pass $pass aborted; stopping. Rerun cc_pilot.sh to resume."
    exit 1
  fi
done

"$PY" scripts/jevlin_bench/analyze.py --out "$OUT_DIR"/pass-*.json \
  --catalog datasets/web-search/coding/tasks/catalog.json \
  --rate-card experiments/rates.json --md "$OUT_DIR/report.md" --json "$OUT_DIR/report.json"
