#!/usr/bin/env python3
"""Summarise one or more harness runs per arm, with the Jevlin-specific measures.

Reads the `--out` JSON files the harness wrote (one per pass) and, for Jevlin
arms, the raw hop dumps under `--raw-dir` (hops/search-NN.response.json), which
carry the full gateway response: which upstream arms ran, what each cost,
whether the cache answered. Direct-vendor cost is an estimate from a rate card
you supply; Jevlin's is the gateway's own metering. The two are never added
into one number without saying so.

Per arm, across passes:
  completion            mean ± SD of pass rate per pass (the board's statistic)
  tasks, passes         how many rows the numbers rest on
  searches/task         logical searches the agent issued
  upstream/task         upstream provider calls (Jevlin: arms not skipped; direct: = searches)
  gold-url retrieved    share of tasks where a gold `# source:` URL appeared in any hop
  search p50/p95 ms     harness-measured search latency (Jevlin: includes its fan-out)
  tokens/task           LLM prompt+completion tokens (median)
  cost/task             Jevlin: observed micro-dollars; direct: rate card × calls (estimate)
  cache hits            Jevlin searches answered from the gateway cache
  errors, empty         hops that raised, hops that returned nothing
  oracle                Jevlin fan-out arms only: tasks where at least one single
                        upstream provider's own results contained a gold URL —
                        the best-single-provider-in-hindsight bound on retrieval

Usage:
  python3 scripts/jevlin_bench/analyze.py --out runs/pass-*.json --raw-dir runs/web-coding \
      --catalog datasets/web-search/coding/tasks/catalog.json [--rate-card rates.json] [--md report.md]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from coding_search.judge import url_was_observed  # noqa: E402

NOT_CALLED = {"skipped"}


def load_rate_card(path: Path | None) -> dict[str, float]:
    """{backend_or_provider: usd_per_search}. Estimates; keep the source in the file."""
    if path is None:
        return {}
    data = json.loads(path.read_text())
    return {str(k): float(v) for k, v in (data.get("usd_per_search") or {}).items()}


def load_catalog_docs(path: Path | None) -> dict[str, list[str]]:
    if path is None:
        return {}
    data = json.loads(path.read_text())
    return {str(item["id"]): list(item.get("docs") or []) for item in data.get("items") or []}


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100) * (len(ordered) - 1))))
    return ordered[index]


def jevlin_hop_bodies(row: dict[str, Any]) -> list[dict[str, Any]]:
    raw_dir = row.get("raw_dir")
    if not raw_dir:
        return []
    bodies: list[dict[str, Any]] = []
    hops = Path(raw_dir) / "hops"
    if not hops.is_dir():
        return []
    for path in sorted(hops.glob("search-*.response.json")):
        try:
            payload = json.loads(path.read_text())
        except ValueError:
            continue
        body = payload.get("body") if isinstance(payload, dict) else None
        if isinstance(body, dict) and "result" in body and "ok" in body:
            body = body.get("result") or {}
        bodies.append(body if isinstance(body, dict) else {})
    return bodies


def jevlin_measures(bodies: list[dict[str, Any]], gold_urls: list[str], max_results: int) -> dict[str, Any]:
    upstream = 0
    cost = 0
    cache_hits = 0
    pending = 0
    providers_seen: set[str] = set()
    oracle_hit = False
    for body in bodies:
        usage = body.get("usage") or {}
        cached = bool(usage.get("cache_hit"))
        if cached:
            cache_hits += 1
        else:
            # A cache hit replays an earlier fan-out: its candidates were not called
            # for this search, and hits are zero-rated, so neither counts here.
            cost += int(usage.get("cost_micros") or 0)
            pending += int(usage.get("pending") or 0)
        for cand in body.get("candidates") or []:
            if not isinstance(cand, dict):
                continue
            status = cand.get("status")
            if status not in NOT_CALLED and not cached:
                upstream += 1
            providers_seen.add(str(cand.get("provider") or ""))
            if status == "ok" and gold_urls:
                own = [str(c.get("url") or "") for c in (cand.get("citations") or [])[:max_results] if isinstance(c, dict)]
                if any(url_was_observed(gold, own) for gold in gold_urls):
                    oracle_hit = True
    return {
        "upstream_calls": upstream,
        "cost_micros": cost,
        "cache_hits": cache_hits,
        "pending": pending,
        "providers": sorted(p for p in providers_seen if p),
        "oracle_hit": oracle_hit,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, nargs="+", required=True, help="harness --out JSON files, one per pass")
    parser.add_argument("--raw-dir", type=Path, default=None, help="unused directly; rows carry raw_dir")
    parser.add_argument("--catalog", type=Path, default=None, help="catalog.json, for gold source URLs")
    parser.add_argument("--rate-card", type=Path, default=None, help='{"usd_per_search": {"exa_fast": 0.005, ...}} (estimates)')
    parser.add_argument("--md", type=Path, default=None, help="write a markdown table here")
    parser.add_argument("--json", type=Path, default=None, help="write the per-arm numbers here")
    args = parser.parse_args(argv)

    rates = load_rate_card(args.rate_card)
    docs = load_catalog_docs(args.catalog)

    per_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, path in enumerate(args.out):
        summary = json.loads(path.read_text())
        max_results = int(summary.get("max_results") or 8)
        for row in summary.get("results") or []:
            arm = str(row.get("search_backend") or row.get("backend") or "")
            metrics = row.get("metrics") or {}
            hops = row.get("search_hops") or []
            gold_urls = docs.get(str(row.get("id")), [])
            observed = [u for hop in hops for u in (hop.get("hit_urls") or [])]
            gold_seen = bool(gold_urls) and any(url_was_observed(g, observed) for g in gold_urls)
            entry = {
                "pass": index,
                "task": row.get("id"),
                "passed": bool(row.get("passed")),
                "run_error": bool(row.get("error")),
                "searches": int(row.get("search_count") or 0),
                "hop_errors": sum(1 for hop in hops if hop.get("error")),
                "hop_empty": sum(1 for hop in hops if hop.get("empty")),
                "search_ms": [int(hop.get("latency_ms") or 0) for hop in hops],
                "tokens": int(metrics.get("llm_prompt_tokens") or 0) + int(metrics.get("llm_completion_tokens") or 0),
                "wall_ms": int(metrics.get("wall_ms") or 0),
                "gold_seen": gold_seen,
                "has_gold": bool(gold_urls),
                "leak": bool(row.get("query_contained_gold_token")),
            }
            if arm.startswith("jevlin"):
                entry.update(jevlin_measures(jevlin_hop_bodies(row), gold_urls, max_results))
                entry["cost_usd"] = entry["cost_micros"] / 1_000_000
                entry["cost_kind"] = "observed"
            else:
                entry["upstream_calls"] = entry["searches"]
                entry["cache_hits"] = 0
                rate = rates.get(arm)
                entry["cost_usd"] = (rate * entry["searches"]) if rate is not None else None
                entry["cost_kind"] = "estimate" if rate is not None else "unknown"
            per_arm[arm].append(entry)

    report: dict[str, Any] = {}
    for arm, rows in sorted(per_arm.items()):
        by_pass: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_pass[row["pass"]].append(row)
        pass_rates = [sum(r["passed"] for r in group) / len(group) for group in by_pass.values() if group]
        search_ms = [ms for r in rows for ms in r["search_ms"]]
        costs = [r["cost_usd"] for r in rows if r.get("cost_usd") is not None]
        gold_rows = [r for r in rows if r["has_gold"]]
        jevlin_rows = [r for r in rows if "oracle_hit" in r]
        report[arm] = {
            "rows": len(rows),
            "passes": len(by_pass),
            "tasks": len({r["task"] for r in rows}),
            "completion_mean": statistics.mean(pass_rates) if pass_rates else None,
            "completion_sd": statistics.stdev(pass_rates) if len(pass_rates) > 1 else None,
            "searches_per_task": statistics.mean(r["searches"] for r in rows) if rows else None,
            "upstream_per_task": statistics.mean(r["upstream_calls"] for r in rows) if rows else None,
            "gold_url_retrieved": (sum(r["gold_seen"] for r in gold_rows) / len(gold_rows)) if gold_rows else None,
            "search_ms_p50": percentile(search_ms, 50),
            "search_ms_p95": percentile(search_ms, 95),
            "tokens_median": statistics.median(r["tokens"] for r in rows) if rows else None,
            "wall_ms_median": statistics.median(r["wall_ms"] for r in rows) if rows else None,
            "cost_usd_per_task": statistics.mean(costs) if costs else None,
            "cost_kind": rows[0]["cost_kind"] if rows else None,
            "cost_usd_per_pass_task": (statistics.mean(costs) / statistics.mean(pass_rates)) if costs and pass_rates and statistics.mean(pass_rates) > 0 else None,
            "cache_hit_searches": sum(r["cache_hits"] for r in rows),
            "pending_arms": sum(r.get("pending", 0) for r in rows),
            "hop_errors": sum(r["hop_errors"] for r in rows),
            "hop_empty": sum(r["hop_empty"] for r in rows),
            "run_errors": sum(r["run_error"] for r in rows),
            "leak_flagged": sum(r["leak"] for r in rows),
            "oracle_single_provider_retrieval": (sum(r["oracle_hit"] for r in jevlin_rows) / len(jevlin_rows)) if jevlin_rows else None,
            "providers_seen": sorted({p for r in jevlin_rows for p in r.get("providers", [])}) if jevlin_rows else None,
        }

    def fmt(value: Any, kind: str = "") -> str:
        if value is None:
            return "–"
        if kind == "pct":
            return f"{100 * value:.1f}%"
        if kind == "usd":
            return f"${value:.4f}"
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)

    lines = [
        "| arm | tasks×passes | completion | searches/task | upstream/task | gold URL seen | search p50/p95 ms | tokens (med) | cost/task | cost/passed | cache hits | errors/empty | oracle |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm, stats in report.items():
        completion = fmt(stats["completion_mean"], "pct")
        if stats["completion_sd"] is not None:
            completion += f" ± {100 * stats['completion_sd']:.1f}"
        cost = fmt(stats["cost_usd_per_task"], "usd")
        if stats["cost_kind"] == "estimate":
            cost += " (est.)"
        lines.append(
            f"| {arm} | {stats['tasks']}×{stats['passes']} | {completion} | {fmt(stats['searches_per_task'])} | "
            f"{fmt(stats['upstream_per_task'])} | {fmt(stats['gold_url_retrieved'], 'pct')} | "
            f"{fmt(stats['search_ms_p50'])}/{fmt(stats['search_ms_p95'])} | {fmt(stats['tokens_median'])} | {cost} | "
            f"{fmt(stats['cost_usd_per_pass_task'], 'usd')} | {stats['cache_hit_searches']} | "
            f"{stats['hop_errors']}/{stats['hop_empty']} | {fmt(stats['oracle_single_provider_retrieval'], 'pct')} |"
        )
    table = "\n".join(lines)
    print(table)
    print("\ncost: Jevlin arms are the gateway's own metering (observed, floor while pending>0); "
          "direct arms are rate card × searches (estimate) or – when no rate card was given. "
          "oracle: share of tasks where one upstream provider alone returned a gold URL; an analytical bound, not a configuration.")
    if args.md:
        args.md.write_text(table + "\n")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
