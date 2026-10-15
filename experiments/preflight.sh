#!/usr/bin/env bash
# Pre-flight before any paid run: one minimal model call, four one-result searches, the CLI arm.
# .env is read by python-dotenv, exactly as the harness reads it. It is never sourced as shell:
# a malformed line (a space after '=') would otherwise run the value as a command and print it.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-}"
[ -z "$PY" ] && { [ -x .venv/bin/python ] && PY=.venv/bin/python || PY=python3; }
PYTHONPATH=scripts exec "$PY" scripts/jevlin_bench/preflight.py
