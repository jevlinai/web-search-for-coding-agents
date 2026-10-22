#!/usr/bin/env python3
"""Evidence funnel: where the gold evidence is lost between upstream providers and a passing file.

Read-only over one or more pass summaries and their raw dumps. Per job, each stage is "any search
in the job":

  upstream recall    some upstream provider's own citations contained a gold `# source:` URL
                     (fan-out arms: candidates[].citations; single-provider arms: the returned hits)
  post-merge recall  the gateway's merged[] contained a gold URL (fan-out arms)
  top-16 visible     a gold URL within the first 16 of the list the arm presents (recomputed with the
                     arm's own presentation: merged order, or the raw round-robin for jevlin_fanout_raw)
  top-8 visible      a gold URL among the hits the model was shown (for jevlin_balanced_split that
                     is the router's whole merged list, up to 40 entries, not the first 8)
  completion         the judge passed the job

then completion conditional on top-8 visibility, both ways. URL matching is the judge's own
(url_was_observed). Gold URLs are the catalog's `docs` (the gold file's `# source:` comments).

  .venv/bin/python scripts/jevlin_bench/evidence_funnel.py --catalog datasets/.../catalog.json \
      runs/cc-pilot/pass-1.json runs/cc-pilot/pass-2.json runs/cc-pilot/pass-3.json [--md out.md]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from coding_search import jevlin  # noqa: E402
from coding_search.judge import url_was_observed  # noqa: E402

SINGLE_PROVIDER_JEVLIN = {"jevlin_fast", "jevlin_single"}


def observed(urls, gold) -> bool:
    return any(url_was_observed(g, urls) for g in gold)


def bodies_for(row) -> list[dict]:
    raw = Path(row.get("raw_dir") or "")
    if not raw.is_absolute():
        raw = ROOT / raw
    out = []
    for path in sorted((raw / "hops").glob("search-*.response.json")):
        try:
            data = json.loads(path.read_text())
            body = (data.get("body") or {}) if isinstance(data, dict) else {}  # a failed hop is dumped as null
        except (ValueError, OSError):
            body = {}
        if isinstance(body, dict) and "result" in body and "ok" in body:  # the CLI envelope
            body = body.get("result") or {}
        out.append(body if isinstance(body, dict) else {})
    return out


def funnel_for_job(row, gold, arm) -> dict:
    shown8 = [h.get("url") or "" for hop in row["search_hops"] for h in hop.get("hits") or []]
    f = {"top8": observed(shown8, gold), "completion": bool(row["passed"])}
    if not arm.startswith("jevlin") or arm in SINGLE_PROVIDER_JEVLIN:
        f.update({"upstream": f["top8"], "merged": f["top8"], "top16": None})
        return f
    up = mg = t16 = False
    for body in bodies_for(row):
        for cand in body.get("candidates") or []:
            if isinstance(cand, dict) and cand.get("status") == "ok":
                if observed([c.get("url") or "" for c in cand.get("citations") or []], gold):
                    up = True
        merged = body.get("merged") if isinstance(body.get("merged"), list) else []
        if observed([m.get("url") or "" for m in merged], gold):
            mg = True
        if arm == "jevlin_fanout_raw":
            first16 = [h["url"] for h in jevlin.present_raw(body, max_results=16)]
        else:
            first16 = [m.get("url") or "" for m in merged[:16]]
        if observed(first16, gold):
            t16 = True
    f.update({"upstream": up, "merged": mg, "top16": t16})
    return f


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("passes", nargs="+", type=Path)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--md", type=Path, default=None)
    args = parser.parse_args(argv)
    gold = {i["id"]: i.get("docs") or [] for i in json.loads(args.catalog.read_text())["items"]}
    per_arm = defaultdict(list)
    for path in args.passes:
        for row in json.loads(path.read_text())["results"]:
            if row.get("error") or not gold.get(row["id"]):
                continue
            per_arm[row["search_backend"]].append(funnel_for_job(row, gold[row["id"]], row["search_backend"]))
    lines = [
        "| arm | n | upstream recall | post-merge recall | top-16 visible | top-8 visible | completion | completion given seen | completion given not seen |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for arm, rows in sorted(per_arm.items()):
        def pct(key):
            vals = [r[key] for r in rows if r[key] is not None]
            return f"{100 * sum(vals) / len(vals):.0f}%" if vals else "–"
        seen = [r for r in rows if r["top8"]]
        unseen = [r for r in rows if not r["top8"]]
        cs = f"{100 * sum(r['completion'] for r in seen) / len(seen):.0f}% ({len(seen)})" if seen else "–"
        cu = f"{100 * sum(r['completion'] for r in unseen) / len(unseen):.0f}% ({len(unseen)})" if unseen else "–"
        lines.append(f"| {arm} | {len(rows)} | {pct('upstream')} | {pct('merged')} | {pct('top16')} | {pct('top8')} | {pct('completion')} | {cs} | {cu} |")
    table = "\n".join(lines)
    if any(jevlin.is_split_arm(arm) for arm in per_arm):
        table += ("\n\nNote: for jevlin_balanced_split the 'top-8 visible' column is every entry the model was shown "
                  "(the router's whole merged list), not the first 8; compare it with the other arms' column accordingly.")
    print(table)
    if args.md:
        args.md.write_text(table + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
