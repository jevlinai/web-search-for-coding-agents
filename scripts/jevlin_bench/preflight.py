#!/usr/bin/env python3
"""Pre-flight before any paid run. Costs one minimal model call and four one-result searches.

Reads .env the same way the harness does (python-dotenv; never sourced as shell, so a
malformed line cannot leak a value), confirms the model and key work, records what the bench
key resolves each tier to, and checks the CLI arm. Prints no secret.

  PYTHONPATH=scripts .venv/bin/python scripts/jevlin_bench/preflight.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from coding_search.env import default_model, llm_transport, load_environment, make_llm_client  # noqa: E402

QUERY = "what is the boiling point of water at sea level"


def model_check() -> bool:
    print("model: one minimal call to the harness model, with the harness's reasoning and tool settings (cents)")
    try:
        from coding_search.agent import REASONING_EFFORT, WEB_SEARCH_TOOL

        model = default_model()
        client = make_llm_client(timeout=60, max_retries=0)
        response = client.responses.create(
            model=model,
            reasoning={"effort": REASONING_EFFORT},
            tools=[WEB_SEARCH_TOOL],
            input=[{"role": "user", "content": "Reply with the single word OK. Do not use any tool."}],
            max_output_tokens=64,
            store=False,
        )
        usage = getattr(response, "usage", None)
        print("  transport=%s model=%s ok output=%r tokens=%s/%s" % (
            llm_transport(), model, (response.output_text or "").strip()[:20],
            getattr(usage, "input_tokens", "?"), getattr(usage, "output_tokens", "?")))
        return True
    except Exception as exc:  # noqa: BLE001
        print("  MODEL CHECK FAILED: %s: %s" % (type(exc).__name__, str(exc)[:300]))
        print("  Fix this before any run: every job dies at its first model call.")
        return False


def tier_checks() -> None:
    key = (os.environ.get("JEVLIN_API_KEY") or "").strip()
    url = (os.environ.get("JEVLIN_API_URL") or "https://api.jevlin.ai").strip().rstrip("/") + "/v1/search"
    print("host:", url)
    if not key:
        print("  JEVLIN_API_KEY missing in .env")
        return
    for tier in ("fast", "balanced", "max"):
        print("tier requested:", tier)
        started = time.perf_counter()
        try:
            response = requests.post(
                url,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"query": QUERY, "tier": tier, "view": "merged", "max_results": 1},
                timeout=60,
            )
        except requests.RequestException as exc:
            print("  request failed:", type(exc).__name__)
            continue
        elapsed = time.perf_counter() - started
        print("  http=%s time=%.2fs" % (response.status_code, elapsed))
        try:
            body = response.json()
        except ValueError:
            print("  ERROR: non-JSON body:", response.text[:300])
            continue
        if "decision" not in body:
            print("  ERROR:", json.dumps(body)[:300])
            continue
        decision, usage = body["decision"], body.get("usage", {})
        arms = [(c.get("provider"), c.get("status")) for c in body.get("candidates", [])]
        print("  tier=%s providers=%s trimmed=%s requested=%s" % (
            decision.get("tier"), decision.get("providers"), decision.get("trimmed"), decision.get("requested", False)))
        print("    usage=%s merged=%d arms=%s" % (usage, len(body.get("merged") or []), arms))


def shadow_check() -> None:
    """Warn when the shell already exports a variable that .env also sets: dotenv keeps the shell's."""
    from dotenv import dotenv_values

    file_values = dotenv_values(ROOT / ".env")
    shadowed = []
    for key, value in file_values.items():
        if not key.startswith(("JEVLIN_", "ANTHROPIC_", "OPENAI_", "LLM_", "CLAUDE_")):
            continue
        live = os.environ.get(key)
        if live is not None and (value or "").strip() and live.strip() != (value or "").strip():
            shadowed.append(key)
    if shadowed:
        print("WARNING: the shell exports %s with a different value than .env; the shell's value is in force." % ", ".join(shadowed))


def limits_check(*, projected_pass_usd: float = 25.0) -> None:
    """What the bench account may spend today, against what one pass of fan-out arms costs.

    The router also enforces per-minute and per-hour request buckets that this endpoint does not
    report; a pass needs on the order of 600 searches an hour across the Jevlin arms. A 429 storm
    in the run is the symptom of those buckets; the operator raises them in the admin console.
    """
    key = (os.environ.get("JEVLIN_API_KEY") or "").strip()
    url = (os.environ.get("JEVLIN_API_URL") or "https://api.jevlin.ai").strip().rstrip("/") + "/v1/limits"
    if not key:
        return
    try:
        response = requests.get(url, headers={"Authorization": f"Bearer {key}"}, timeout=30)
        body = response.json()
    except (requests.RequestException, ValueError) as exc:
        print("limits: request failed:", type(exc).__name__)
        return
    if response.status_code != 200:
        print("limits: http=%s %s" % (response.status_code, json.dumps(body)[:200]))
        return
    today = body.get("today") or {}
    ceiling = int(today.get("ceiling_micros") or 0) / 1e6
    remaining = int(today.get("remaining_micros") or 0) / 1e6
    spent = int(today.get("spent_micros") or 0) / 1e6
    print("limits: today's ceiling $%.2f, spent $%.2f, remaining $%.2f" % (ceiling, spent, remaining))
    if remaining < projected_pass_usd:
        print("  WARNING: one pass of the Jevlin fan-out arms needs about $%.0f; the account will hit its daily ceiling (402) mid-pass. "
              "Raise the ceiling for the bench project before running." % projected_pass_usd)
    print("  NOTE: per-minute and per-hour request caps are not reported here; the pilot needs ~600 Jevlin searches/hour. "
          "If the smoke or the first jobs show 429s, raise them in the admin console.")


def cli_check() -> None:
    exe = (os.environ.get("JEVLIN_BIN") or "").strip()
    if not exe or not os.access(exe, os.X_OK):
        print("cli: JEVLIN_BIN not set or not executable; run experiments/jevlin-cli/setup.sh first")
        return
    from coding_search.jevlin import JevlinCli

    arm = JevlinCli()
    if not arm.config_path():
        print("cli: WARNING JEVLIN_CLI_CONFIG is not set; the arm would use whatever config the environment implies")
    print("cli: %s  config=%s" % (exe, arm.config_path() or "<inherited>"))
    try:
        print("  " + subprocess.run([exe, "version"], capture_output=True, text=True, timeout=30, check=False).stdout.strip())
        proc = subprocess.run(
            arm.argv(),
            input=json.dumps({"version": 1, "query": QUERY, "max_results": 1}),
            capture_output=True, text=True, timeout=90, check=False, env=arm.env(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print("  cli failed:", type(exc).__name__)
        return
    try:
        envelope = json.loads(proc.stdout or "{}")
    except ValueError:
        print("  ERROR: non-JSON stdout; exit=%s stderr=%s" % (proc.returncode, (proc.stderr or "")[:300]))
        return
    result = envelope.get("result") or {}
    print("  ok=%s http=%s decision=%s usage=%s merged=%d mining=%s error=%s" % (
        envelope.get("ok"), envelope.get("http_status"), result.get("decision"), result.get("usage"),
        len(result.get("merged") or []), envelope.get("mining"), envelope.get("error")))


def main() -> int:
    load_environment()
    shadow_check()
    ok = model_check()
    limits_check()
    tier_checks()
    cli_check()
    print()
    print("Set JEVLIN_PROVIDERS in .env to exactly the balanced providers list above (comma-separated).")
    print("If max resolved to the same set as balanced, drop jevlin_max from ARMS.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
