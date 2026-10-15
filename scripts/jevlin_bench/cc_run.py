#!/usr/bin/env python3
"""Run the coding-search tickets with headless Claude Code as the agent.

Same tickets, judge, budgets, hit shape and output schema as `coding_search run`; a different
agent. Each (ticket, arm) job is one `claude -p` session under the user's subscription, started
with an empty environment (no API key, no JEVLIN_*, no hooks, skills or MCP), whose only
permitted tool is one generated shell command, `web_search "<query>"`, that calls the harness
backend for the arm (cc_search.py) and records every hop. The session's final message is judged
with the harness's own compile + gold-token + grounded-URL check against those hops.

Output: a summary JSON with the same rows `coding_search run` writes (so analyze.py works
unchanged) plus an `agent` block per row (Claude Code version, model, session id, turns, API-
equivalent cost), and raw dumps in the harness's rawlog layout.

  PYTHONPATH=scripts .venv/bin/python scripts/jevlin_bench/cc_run.py \
      --arms perplexity_low,exa_fast,jevlin_balanced --ids hard_sn_activate_shared_snapshot \
      --seed 1 --out runs/cc-smoke/pass-1.json --raw-dir runs/cc-smoke/raw --jobs-dir runs/cc-smoke/jobs

A subscription window that is exhausted shows up as a Claude Code error; the driver waits
--rate-limit-wait seconds and retries the same job up to --rate-limit-retries times before
counting it as a run-level failure. --resume-from and --abort-after-run-errors behave as in
`coding_search run`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from coding_search import agent as harness_agent  # noqa: E402
from coding_search import rawlog  # noqa: E402
from coding_search.env import load_environment  # noqa: E402
from coding_search.eval import load_resume_rows  # noqa: E402
from coding_search.judge import extract_code, hop_retrieved_text, hop_urls, score_needles  # noqa: E402
from coding_search.search import DEFAULT_MAX_RESULTS, resolve_backend  # noqa: E402
from coding_search.tasks import load_tasks  # noqa: E402

DEFAULT_MODEL = "claude-sonnet-5-5"
RATE_LIMIT_RE = re.compile(r"rate.?limit|usage limit|limit reached|hit your limit|resets at|too many requests|overloaded|\b529\b", re.I)

# The harness's search-only system prompt with its tool sentences rewritten for a shell command.
# The citation and stopping rules are the harness's words, unchanged. Identical for every arm.
SYSTEM_PROMPT = """You edit an existing production script. Make the smallest change that satisfies the ticket.
Do not rewrite the file from scratch. Use the latest public APIs. Do not invent symbols that are not in the current docs.
Your only tool is the shell command `{command} "<query>"`, run with Bash exactly as written (full path, one quoted argument). It returns JSON: {{"backend", "query", "hits": [{{"url", "title", "snippet"}}]}}. You must call it before you return the file. Do not rely on memory for params, paths, or versions. Search for the current docs. If snippets are not enough, search again with a more specific query. You may call it at most {max_searches} times; after that it answers "search budget exhausted; answer now". You have no other tool: do not run any other command, and do not read, write or fetch anything.
For every API version or field you add, put a `# source: https://...` comment on the line above with the search-result URL where you saw it. Do not use a name you cannot cite from a result you received.
Do not return until you are sure the path, version, and field names match current docs. If you are unsure, search again. Using more of the search budget is better than returning a wrong file. Only stop early when you are sure. Return only the full updated file, in a single ```python fenced block, as your final message."""


def resolve_claude(claude_bin: str) -> str:
    """Absolute path of the launcher, found on the parent's PATH once; the session gets no such PATH."""
    found = shutil.which(claude_bin) if not os.path.isabs(claude_bin) else claude_bin
    return os.path.abspath(found or claude_bin)


def claude_version(claude_bin: str) -> str:
    try:
        return subprocess.run([resolve_claude(claude_bin), "--version"], capture_output=True, text=True, timeout=30, check=False).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def clean_env(claude_bin: str, tools_dir: Path | None = None) -> dict[str, str]:
    """The session's environment: enough to run claude and the wrapper, nothing else.

    No ANTHROPIC_API_KEY (the subscription login must be used), no JEVLIN_* (the wrapper loads
    .env itself), no CLAUDECODE markers from a parent session, no hooks or skills (--safe-mode).
    The job's tools directory leads PATH so the model can type `web_search "..."` by name.
    """
    # The directory of the `claude` launcher as found on PATH, not resolved through its symlink:
    # the symlink points into the npm package, while `node` lives beside the launcher.
    claude_dir = str(Path(resolve_claude(claude_bin)).parent)
    dirs = ([str(tools_dir)] if tools_dir else []) + [claude_dir, "/usr/bin", "/bin", "/usr/sbin", "/sbin"]
    path = ":".join(dict.fromkeys(dirs))
    env = {
        "HOME": os.environ.get("HOME", ""),
        "USER": os.environ.get("USER", ""),
        "LOGNAME": os.environ.get("LOGNAME", os.environ.get("USER", "")),
        "SHELL": "/bin/sh",
        "TERM": "xterm-256color",
        "LANG": os.environ.get("LANG", "en_US.UTF-8"),
        "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
        "PATH": path,
        "DISABLE_AUTOUPDATER": "1",
    }
    return {k: v for k, v in env.items() if v}


def write_wrapper(tools_dir: Path, *, arm: str, hop_dir: Path, max_searches: int, max_results: int, python: str, fixture: str | None) -> Path:
    tools_dir.mkdir(parents=True, exist_ok=True)
    wrapper = tools_dir / "web_search"
    lines = [
        "#!/bin/sh",
        f"export CC_ROOT='{ROOT}'",
        f"export CC_ARM='{arm}'",
        f"export CC_HOP_DIR='{hop_dir}'",
        f"export CC_MAX_SEARCHES='{max_searches}'",
        f"export CC_MAX_RESULTS='{max_results}'",
    ]
    if fixture:
        lines.append(f"export CC_FIXTURE_PATH='{fixture}'")
    lines.append(f"exec '{python}' '{HERE / 'cc_search.py'}' \"$@\"")
    wrapper.write_text("\n".join(lines) + "\n")
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return wrapper


def claude_argv(claude_bin: str, *, prompt: str, system: str, wrapper: Path, model: str) -> list[str]:
    return [
        resolve_claude(claude_bin), "-p", prompt,
        "--model", model,
        "--output-format", "json",
        "--append-system-prompt", system,
        "--tools", "Bash",
        # Both spellings of the one permitted command: the absolute path (first, so tooling that
        # reads the pattern finds the file) and the bare name the prompt uses, which resolves
        # through the session PATH to the same file.
        "--allowedTools", f"Bash({wrapper} *)", f"Bash({wrapper.name} *)",
        "--disallowedTools", "WebSearch,WebFetch,Read,Write,Edit,Glob,Grep,Task,TodoWrite,NotebookEdit",
        "--permission-mode", "dontAsk",
        "--permission-prompts", "none",
        "--safe-mode",
        "--strict-mcp-config",
        "--disable-slash-commands",
        "--no-session-persistence",
    ]


def parse_result(stdout: str) -> dict[str, Any]:
    text = (stdout or "").strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
    except ValueError:
        # stream or noise before the object: take the last JSON object line
        for line in reversed(text.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    data = json.loads(line)
                    break
                except ValueError:
                    continue
        else:
            return {"_unparsed": text[:2000]}
    if isinstance(data, list):
        results = [item for item in data if isinstance(item, dict) and item.get("type") == "result"]
        data = results[-1] if results else (data[-1] if data and isinstance(data[-1], dict) else {})
    return data if isinstance(data, dict) else {}


def is_rate_limited(result: dict[str, Any], stderr: str, returncode: int) -> bool:
    text = " ".join(str(result.get(k) or "") for k in ("result", "error", "subtype", "_unparsed")) + " " + (stderr or "")
    return bool(result.get("is_error") or returncode != 0) and bool(RATE_LIMIT_RE.search(text))


def usage_tokens(result: dict[str, Any]) -> tuple[int, int]:
    usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
    prompt = int(usage.get("input_tokens") or 0) + int(usage.get("cache_creation_input_tokens") or 0) + int(usage.get("cache_read_input_tokens") or 0)
    completion = int(usage.get("output_tokens") or 0)
    return prompt, completion


def load_hops(hop_dir: Path, leak_needles: list[str]) -> list[dict[str, Any]]:
    hops = []
    for path in sorted(hop_dir.glob("search-*.json")):
        try:
            hop = json.loads(path.read_text())
        except ValueError:
            continue
        hop["gold_token"] = harness_agent._hidden_token_in_query(str(hop.get("query") or ""), leak_needles)
        hops.append(hop)
    return hops


def run_job(task, arm: str, *, args, jobs_dir: Path, claude_bin: str, version: str, fixture: str | None, run_stamp: str) -> dict[str, Any]:
    label = f"{arm}+search"
    job_dir = jobs_dir / task.id / arm
    if job_dir.exists():
        shutil.rmtree(job_dir)
    hop_dir = job_dir / "hops"
    hop_dir.mkdir(parents=True)
    # One command path for the whole run, rewritten before each job with that job's arm and hop
    # directory. The model is told the absolute path: Claude Code's Bash runs through the user's
    # shell, which rebuilds PATH from their profile, so a bare name is not reliably found.
    wrapper = write_wrapper(jobs_dir / "bin", arm=arm, hop_dir=hop_dir, max_searches=args.max_searches,
                            max_results=args.max_results, python=sys.executable, fixture=fixture)
    system = SYSTEM_PROMPT.format(max_searches=args.max_searches, command=wrapper)
    (job_dir / "prompt.txt").write_text(task.prompt)
    (job_dir / "system.txt").write_text(system)
    argv = claude_argv(claude_bin, prompt=task.prompt, system=system, wrapper=wrapper, model=args.model)
    env = clean_env(claude_bin, tools_dir=wrapper.parent)
    attempts = 0
    waited_s = 0
    started = time.perf_counter()
    result: dict[str, Any] = {}
    stderr = ""
    returncode = 0
    run_error: str | None = None
    while True:
        attempts += 1
        try:
            proc = subprocess.run(argv, cwd=job_dir, env=env, capture_output=True, text=True, timeout=args.job_timeout, check=False)
            stdout, stderr, returncode = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired:
            stdout, stderr, returncode = "", f"claude timed out after {args.job_timeout}s", 124
        except OSError as exc:
            stdout, stderr, returncode = "", f"{type(exc).__name__}: {exc}", 127
        (job_dir / f"claude.attempt-{attempts}.stdout.json").write_text(stdout or "")
        (job_dir / f"claude.attempt-{attempts}.stderr.txt").write_text(stderr or "")
        result = parse_result(stdout)
        if is_rate_limited(result, stderr, returncode) and attempts <= args.rate_limit_retries:
            print(f"  rate limited on {task.id} {arm}; waiting {args.rate_limit_wait}s (attempt {attempts})", flush=True)
            for path in hop_dir.glob("*"):
                path.unlink()
            time.sleep(args.rate_limit_wait)
            waited_s += args.rate_limit_wait
            continue
        break
    wall_ms = int((time.perf_counter() - started) * 1000) - waited_s * 1000
    if returncode != 0 and not result:
        run_error = f"claude exit {returncode}: {(stderr or '').strip()[:500]}"
    elif result.get("is_error"):
        run_error = f"claude error ({result.get('subtype')}): {str(result.get('result') or result.get('error') or '')[:500]}"
    elif not result:
        run_error = "claude produced no JSON result"

    hops = load_hops(hop_dir, task.leak_needles())
    answer = str(result.get("result") or "") if not run_error else ""
    source = extract_code(answer, task.outfile)
    run_like = {"searches": hops, "fetches": []}
    score = score_needles(source, task, observed_urls=hop_urls(run_like), retrieved_text=hop_retrieved_text(run_like))
    prompt_tokens, completion_tokens = usage_tokens(result)
    notes: list[str] = []
    if (hop_dir / "budget_exhausted").exists():
        notes.append("search budget exhausted")
    if not hops:
        notes.append("no search was made")
    denials = result.get("permission_denials") if isinstance(result.get("permission_denials"), list) else []
    for denial in denials[:10]:
        notes.append(f"tool denied: {json.dumps(denial, default=str)[:160]}")
    if result.get("subtype") and result.get("subtype") != "success":
        notes.append(f"claude subtype: {result.get('subtype')}")
    for hop in hops:
        if hop.get("gold_token"):
            notes.append(f"query contained gold token:{hop['gold_token']}")
    if run_error:
        notes.append(run_error)
    metrics = {
        "wall_ms": wall_ms,
        "llm_ms": int(result.get("duration_api_ms") or 0),
        "search_count": len(hops),
        "fetch_count": 0,
        "search_ms": sum(int(h.get("latency_ms") or 0) for h in hops),
        "fetch_ms": 0,
        "llm_prompt_tokens": prompt_tokens,
        "llm_completion_tokens": completion_tokens,
        "gold_token_queries": sum(1 for h in hops if h.get("gold_token")),
    }
    row: dict[str, Any] = {
        "id": task.id,
        "backend": label,
        "search_backend": arm,
        "allow_fetch": False,
        "used_search": bool(hops),
        "used_fetch": False,
        "search_count": len(hops),
        "fetch_count": 0,
        "queries": [str(h.get("query") or "") for h in hops],
        "fetch_urls": [],
        "search_hops": [{k: v for k, v in h.items() if k not in ("request", "response", "vendor_meta")} for h in hops],
        "fetch_hops": [],
        "passed": bool(score.passed) and not run_error,
        "score": score.to_dict() if not run_error else {"passed": False, "compiled": False, "output": "RUN_ERROR", "notes": [run_error]},
        "metrics": metrics,
        "source": source,
        "notes": notes,
        "braintrust_permalink": None,
        "query_contained_gold_token": any(h.get("gold_token") for h in hops),
        "error": run_error,
        "agent": {
            "kind": "claude-code",
            "version": version,
            "model": args.model,
            "session_id": result.get("session_id"),
            "num_turns": result.get("num_turns"),
            "duration_ms": result.get("duration_ms"),
            "duration_api_ms": result.get("duration_api_ms"),
            "total_cost_usd": result.get("total_cost_usd"),
            "subtype": result.get("subtype"),
            "attempts": attempts,
            "rate_limit_wait_s": waited_s,
            "model_usage": result.get("modelUsage"),
            "job_dir": str(job_dir),
        },
    }
    if args.raw_dir is not None:
        calls = [SimpleNamespace(request=h.get("request"), response=h.get("response")) for h in hops]
        dest = rawlog.write_arm_raw(
            args.raw_dir, dataset="", split="search-only", backend=arm, allow_fetch=False, task_id=task.id,
            run=SimpleNamespace(searches=calls, fetches=[], notes=notes), row=row, run_stamp=run_stamp,
        )
        row["raw_dir"] = str(dest)
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", required=True, help="comma list of backends (harness vendors and Jevlin arms)")
    parser.add_argument("--ids", default="", help="comma list of ticket ids; default all")
    parser.add_argument("--seed", type=int, default=None, help="shuffle the (ticket, arm) order with this seed")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--jobs-dir", type=Path, default=None, help="per-job work dirs (wrapper, hops, claude output); default <out dir>/jobs")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-searches", type=int, default=5)
    parser.add_argument("--max-results", type=int, default=DEFAULT_MAX_RESULTS)
    parser.add_argument("--job-timeout", type=int, default=900, help="seconds per Claude Code session")
    parser.add_argument("--rate-limit-wait", type=int, default=1200)
    parser.add_argument("--rate-limit-retries", type=int, default=12)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--abort-after-run-errors", type=int, default=5)
    parser.add_argument(
        "--abort-after-hop-errors",
        type=int,
        default=12,
        help="Stop the pass after this many consecutive failed searches across jobs (a vendor or gateway refusing every call, e.g. 429); 0 disables.",
    )
    parser.add_argument("--claude-bin", default="claude")
    parser.add_argument("--fixture", default=None, help="tests only: merged-response fixture served to the jevlin_fixture arm")
    parser.add_argument("--run-stamp", default="")
    args = parser.parse_args(argv)

    load_environment()
    # Absolute paths throughout: the session's working directory is the job directory, the model is
    # told the command's path, and the allow-rule is built from the same string.
    args.out = args.out.resolve()
    if args.raw_dir is not None:
        args.raw_dir = args.raw_dir.resolve()
    args.jobs_dir = (args.jobs_dir or (args.out.parent / "jobs")).resolve()
    if args.resume_from is not None:
        args.resume_from = args.resume_from.resolve()
    ids = [item.strip() for item in args.ids.split(",") if item.strip()]
    tasks = load_tasks(ids=ids or None)
    arms = [resolve_backend(item.strip()) for item in args.arms.split(",") if item.strip()]
    run_stamp = rawlog.parse_run_stamp(args.run_stamp) or rawlog.new_run_stamp()
    jobs_dir = args.jobs_dir
    version = claude_version(args.claude_bin)
    jobs = [(task, arm) for task in tasks for arm in arms]
    if args.seed is not None:
        random.Random(args.seed).shuffle(jobs)
    resumed_rows, done_keys = load_resume_rows(args.resume_from)
    if done_keys:
        before = len(jobs)
        jobs = [job for job in jobs if (job[0].id, job[1], False) not in done_keys]
        print(f"resume: keeping {len(resumed_rows)} rows from {args.resume_from}; {before - len(jobs)} jobs skipped, {len(jobs)} to run", flush=True)
    print(f"agent=claude-code version={version} model={args.model} arms={arms} n={len(jobs)} seed={args.seed} run_stamp={run_stamp}", flush=True)

    results: list[dict[str, Any]] = list(resumed_rows)
    aborted = False
    consecutive = 0
    hop_errors_in_a_row = 0
    started = datetime.now(timezone.utc)
    for task, arm in jobs:
        row = run_job(task, arm, args=args, jobs_dir=jobs_dir, claude_bin=args.claude_bin, version=version, fixture=args.fixture, run_stamp=run_stamp)
        results.append(row)
        hop_errs = sum(1 for h in row["search_hops"] if h.get("error"))
        print(task.id, row["backend"], "pass" if row["passed"] else "fail", (row.get("error") or "")[:160],
              f"[{hop_errs}/{len(row['search_hops'])} search errors]" if hop_errs else "", flush=True)
        consecutive = consecutive + 1 if row.get("error") else 0
        if args.abort_after_run_errors and consecutive >= args.abort_after_run_errors:
            print(f"ABORTING PASS: {consecutive} consecutive run-level failures; last: {str(row.get('error') or '')[:200]}", flush=True)
            aborted = True
            break
        # Failed searches are data for one job; a run of them across jobs is a vendor or gateway
        # refusing every call (a 429 storm), and every further job would be judged on nothing.
        for hop in row["search_hops"]:
            hop_errors_in_a_row = hop_errors_in_a_row + 1 if hop.get("error") else 0
        if args.abort_after_hop_errors and hop_errors_in_a_row >= args.abort_after_hop_errors:
            last = next((h.get("error") for h in reversed(row["search_hops"]) if h.get("error")), "")
            print(f"ABORTING PASS: {hop_errors_in_a_row} consecutive failed searches; last: {str(last)[:200]}", flush=True)
            aborted = True
            break
        args.out.parent.mkdir(parents=True, exist_ok=True)
        _write_summary(args, results, arms, version, run_stamp, aborted=False, partial=True, started=started, planned=len(jobs) + len(resumed_rows))
    _write_summary(args, results, arms, version, run_stamp, aborted=aborted, partial=False, started=started, planned=len(jobs) + len(resumed_rows))
    passed = sum(1 for row in results if row["passed"])
    print(f"passed={passed}/{len(results)} wrote {args.out}", flush=True)
    return 1 if aborted else 0


def _write_summary(args, results, arms, version, run_stamp, *, aborted, partial, started, planned) -> None:
    results = sorted(results, key=lambda row: (row["id"], row["backend"]))
    passed = sum(1 for row in results if row["passed"])
    summary = {
        "n": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "experiment": f"claude-code/search-only/{'+'.join(arms)}",
        "experiment_url": None,
        "backend": list(arms),
        "fetch_mode": "off",
        "model": args.model,
        "agent": {"kind": "claude-code", "version": version,
                  "system_prompt": SYSTEM_PROMPT.format(max_searches=args.max_searches, command=str(args.jobs_dir / "bin" / "web_search"))},
        "concurrency": 1,
        "max_results": args.max_results,
        "max_searches": args.max_searches,
        "shuffle_seed": args.seed,
        "resumed_from": str(args.resume_from) if args.resume_from else None,
        "aborted": aborted,
        "partial": partial,
        "planned": planned,
        "started_at": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "written_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "raw_dir": str(args.raw_dir) if args.raw_dir else None,
        "run_stamp": run_stamp,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, default=str) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
