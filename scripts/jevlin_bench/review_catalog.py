#!/usr/bin/env python3
"""Review the derived gold tokens against the documentation they are supposed to come from.

For every ticket, fetch the gold `# source:` page(s) and check each `required` token:
  exact        the token appears verbatim in the page (or in the ticket text)
  case-only    it appears only case-insensitively — the public gold's casing differs from the
               documentation's; an agent following the docs can never satisfy the judge
  absent       it does not appear at all (sample data, or a page that needs JavaScript / login)
  unavailable  the page could not be fetched

With --apply, every `case-only` token is rewritten to the documentation's casing in
experiments/catalog-corrections.json (which build_public_catalog.py applies to both the catalog
and the gold file). Nothing else is changed automatically; `absent` tokens are listed for a human.

  .venv/bin/python scripts/jevlin_bench/review_catalog.py [--apply] [--out datasets]
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CORRECTIONS = ROOT / "experiments" / "catalog-corrections.json"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"


def fetch(url: str) -> str | None:
    try:
        response = requests.get(url, headers={"User-Agent": UA, "Accept": "text/html,application/json;q=0.9,*/*;q=0.8"}, timeout=25)
    except requests.RequestException:
        return None
    if response.status_code >= 400:
        return None
    text = response.text
    # Entities and escaped JSON so camelCase in embedded data is visible.
    text = html.unescape(text).replace("\\/", "/")
    return text


def classify(token: str, haystacks: list[str]) -> tuple[str, str | None]:
    for hay in haystacks:
        if token in hay:
            return "exact", None
    for hay in haystacks:
        match = re.search(re.escape(token), hay, re.IGNORECASE)
        if match:
            return "case-only", match.group(0)
    return "absent", None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "datasets")
    parser.add_argument("--apply", action="store_true", help="write case-only corrections to experiments/catalog-corrections.json")
    args = parser.parse_args(argv)
    tasks_dir = args.out / "web-search" / "coding" / "tasks"
    catalog = json.loads((tasks_dir / "catalog.json").read_text())
    corrections = json.loads(CORRECTIONS.read_text()) if CORRECTIONS.is_file() else {"version": 1, "tokens": {}}
    review: list[dict] = []
    pages: dict[str, str | None] = {}
    for item in catalog["items"]:
        tid = item["id"]
        ticket = (tasks_dir / tid / "prompt.txt").read_text()
        haystacks = []
        fetched = []
        for url in item.get("docs") or []:
            if url not in pages:
                pages[url] = fetch(url)
            fetched.append((url, pages[url] is not None, len(pages[url] or "")))
            if pages[url]:
                haystacks.append(pages[url])
        verdicts = []
        for token in item["required"]:
            if any(token in h for h in haystacks) or token in ticket:
                verdicts.append({"token": token, "status": "exact"})
                continue
            status, found = classify(token, haystacks) if haystacks else ("unavailable", None)
            entry = {"token": token, "status": status}
            if found:
                entry["documented_as"] = found
            verdicts.append(entry)
        review.append({"task_id": tid, "pages": fetched, "tokens": verdicts})
        flags = [f"{v['token']}→{v.get('documented_as')}" for v in verdicts if v["status"] == "case-only"]
        absent = [v["token"] for v in verdicts if v["status"] in ("absent", "unavailable")]
        print(f"{tid:46} pages={[(ok, n) for _, ok, n in fetched]} case-only={flags} absent={absent}")
        if args.apply:
            for v in verdicts:
                if v["status"] == "case-only":
                    corrections["tokens"].setdefault(tid, {})[v["token"]] = v["documented_as"]
    (tasks_dir / "review.json").write_text(json.dumps(review, indent=2) + "\n")
    if args.apply:
        corrections["note"] = ("Tokens whose casing in the public gold differs from the documentation the gold cites. "
                               "Applied by build_public_catalog.py to both catalog.json and the gold file. Found by review_catalog.py.")
        CORRECTIONS.write_text(json.dumps(corrections, indent=2) + "\n")
        print(f"wrote {CORRECTIONS} with {sum(len(v) for v in corrections['tokens'].values())} token corrections")
    totals = {}
    for row in review:
        for v in row["tokens"]:
            totals[v["status"]] = totals.get(v["status"], 0) + 1
    print("totals:", totals)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
