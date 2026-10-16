#!/usr/bin/env python3
"""Rescore: what the fan-out's own raw dumps would have shown under other presentation rules.

No agent runs. For every fan-out job in the pass summaries, each search's dumped gateway body
(candidates[].citations, merged[]) is re-presented under alternative ORDERING rules and
alternative EXCERPT rules, and two things are measured per job, "any search in the job":

  gold in top-8     a gold `# source:` URL (catalog `docs`) is among the first 8 hits
  token in top-8    AND the hit's shown excerpt contains one of the ticket's `required` tokens,
                    which is what lets the judge ground the token without a fetch

URL matching is the judge's own (url_was_observed). The current production rule is
`votes`+`longest`; the table shows how far each alternative moves each number, over the same
dumps, so a change is argued from a ceiling before it is argued from a rerun.

  python3 scripts/jevlin_bench/rescore.py --catalog datasets/.../catalog.json \
      runs/cc-pilot/pass-1.json runs/cc-pilot/pass-2.json runs/cc-pilot/pass-3.json [--md out.md]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from coding_search.judge import url_was_observed  # noqa: E402

FANOUT_ARMS = {"jevlin_balanced", "jevlin_fanout_merged", "jevlin_fanout_raw", "jevlin_cli"}
TOP = 8
SNIPPET_CAP = 1200
RRF_K = 60

# ---------------------------------------------------------------- reading the dumps

def bodies_for(row) -> list[dict]:
    raw = Path(row.get("raw_dir") or "")
    if not raw.is_absolute():
        raw = ROOT / raw
    out = []
    for path in sorted((raw / "hops").glob("search-*.response.json")):
        try:
            data = json.loads(path.read_text())
            body = (data.get("body") or {}) if isinstance(data, dict) else {}
        except (ValueError, OSError):
            body = {}
        if isinstance(body, dict) and "result" in body and "ok" in body:
            body = body.get("result") or {}
        out.append(body if isinstance(body, dict) else {})
    return out


def pages(body) -> list[dict]:
    """One record per page across the providers that answered: url, and every provider's listing
    (provider, rank, snippet). Dispatch order is the candidates' order."""
    at, out = {}, []
    for cand in body.get("candidates") or []:
        if not isinstance(cand, dict) or cand.get("status") != "ok":
            continue
        seen = set()
        for cite in cand.get("citations") or []:
            url = cite.get("url") or ""
            if not url or url in seen:
                continue
            seen.add(url)
            if url not in at:
                at[url] = len(out)
                out.append({"url": url, "title": cite.get("title") or "", "listings": []})
            out[at[url]]["listings"].append({
                "provider": cand.get("provider") or "", "rank": int(cite.get("rank") or 99),
                "snippet": cite.get("snippet") or "", "excerpts": [e for e in (cite.get("excerpts") or []) if e],
            })
    return out

# ---------------------------------------------------------------- ordering rules

def order_votes(ps):      # production: providers that found it, then best rank
    return sorted(ps, key=lambda p: (-len(p["listings"]), min(l["rank"] for l in p["listings"])))

def order_rrf(ps):        # reciprocal rank fusion
    return sorted(ps, key=lambda p: -sum(1.0 / (RRF_K + l["rank"]) for l in p["listings"]))

def order_bestrank(ps):   # best position first, votes break ties
    return sorted(ps, key=lambda p: (min(l["rank"] for l in p["listings"]), -len(p["listings"])))

def order_roundrobin(ps): # jevlin_fanout_raw's presentation: every provider's 1st, then 2nd, ...
    return sorted(ps, key=lambda p: (min(l["rank"] for l in p["listings"]), ps.index(p)))

ORDERINGS = {"votes (prod)": order_votes, "rrf": order_rrf, "best-rank": order_bestrank, "round-robin": order_roundrobin}

# Optional fifth ordering: TypeSafe Jev scores every page on webctl's 4-level rubric against the
# query and the ticket; pages are ordered by expected level. Needs JEV_API_KEY (or
# ~/.config/search-bench/typesafe.key); every answer is cached in --jev-cache so reruns are free.
JEV_LEVELS = [
    "Off-topic: about a different subject, or a different meaning of the query's words",
    "Low value: an SEO, affiliate, or content-farm page, a product listing, aggregator, or vendor pitch, a thin listicle, or a page that only mentions the topic",
    "Useful: substantive, accurate treatment from an established source - a reference work or encyclopedia, an established publication or institution, a vendor's documentation, a practitioner's write-up, or a Q&A with real answers",
    "Best available: a primary source, official docs, original research, or first-hand practitioner experience - the page an expert would open first for this query",
]
JEV_INSTR = ("The state holds a search query (`request`), the task the searcher is working on (`goal`) and a list of "
             "search results. Score `results[{i}]` and judge only that one result: whether opening this page would move "
             "the searcher toward `goal`, and whether it is a source worth reading. Official documentation and primary "
             "sources outrank blogs and listicles; a page that merely matches the query's keywords is not enough.")
_jev_cache = {}
_jev_cache_path = None

def jev_key():
    import os
    k = os.environ.get("JEV_API_KEY") or os.environ.get("TYPESAFE_API_KEY")
    if not k:
        f = Path.home() / ".config" / "search-bench" / "typesafe.key"
        k = f.read_text().strip() if f.is_file() else ""
    return k

def jev_scores(request, goal, ps):
    import hashlib, time, urllib.request, urllib.error
    key = hashlib.sha256(json.dumps([request, goal[:2000], [p["url"] for p in ps]]).encode()).hexdigest()
    if key in _jev_cache:
        return _jev_cache[key]
    results = [{"source": p["url"].split("/")[2] if "://" in p["url"] else "", "title": p["title"], "snippet": ex_longest(p)[:600]} for p in ps]
    state = {"request": request, "goal": goal[:2000], "results": results}
    scores = []
    for b in range(0, len(ps), 40):
        qs = {f"s{i}": {"type": "score", "instructions": JEV_INSTR.format(i=i), "criteria": JEV_LEVELS} for i in range(b, min(b + 40, len(ps)))}
        req = urllib.request.Request("https://api.typesafe.ai/v1/systemone", json.dumps({"state": state, "model": "jev-latest", "questions": qs}).encode(),
                                     {"Authorization": f"Bearer {jev_key()}", "Content-Type": "application/json"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    ans = json.load(r)["answers"]
                break
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
                if attempt == 4:
                    raise
                time.sleep(2 * (attempt + 1))
        for i in range(b, min(b + 40, len(ps))):
            scores.append(sum(float(k) * v for k, v in ans[f"s{i}"]["probabilities"].items()))
    _jev_cache[key] = scores
    if _jev_cache_path:
        _jev_cache_path.write_text(json.dumps(_jev_cache))
    return scores

def make_order_jev(request, goal):
    def order(ps):
        if not ps:
            return ps
        sc = jev_scores(request, goal, ps)
        return [p for _, p in sorted(zip(sc, ps), key=lambda t: (-round(t[0], 2), -len(t[1]["listings"]), min(l["rank"] for l in t[1]["listings"])))]
    return order

# Query-aware orderings (added 2026-10-16 after the first rescore): no model call. A page is scored
# by how many of the *query's own* tokens appear in its title, snippets and excerpts — the signal
# every vote/rank rule above ignores — and ties keep round-robin order. Two token sets: the query's
# identifiers only (paths, camelCase, snake_case, backticks), or identifiers plus every word of four
# letters or more that is not a stop word. A query with no usable token falls back to round-robin.
STOP = {"what", "with", "from", "that", "this", "does", "into", "when", "which", "where", "using", "about", "your", "have", "been",
        "will", "should", "would", "could", "their", "there", "them", "then", "than", "also", "only", "each", "such", "other",
        "some", "more", "most", "very", "just", "like", "need", "want", "make", "made", "find", "show", "between", "through",
        "after", "before", "while", "because", "example", "examples", "documentation", "docs", "guide", "tutorial", "reference"}
WORD = re.compile(r"[A-Za-z][A-Za-z0-9_.\-/]{3,}")

def query_tokens(query, idents_only):
    # whole matches, not findall's group captures (IDENT has a capturing group, so findall would return
    # the path segment or an empty string instead of the identifier)
    toks = [m.group(0).lower().strip("`") for m in IDENT.finditer(query)]
    if not idents_only:
        toks += [w.lower() for w in WORD.findall(query) if len(w) >= 4 and w.lower() not in STOP]
    return list(dict.fromkeys(t for t in toks if len(t) >= 3))

def page_text(p):
    return " ".join([p["title"], *[l["snippet"] for l in p["listings"]], *[e for l in p["listings"] for e in l["excerpts"]]]).lower()

def make_order_tokens(query, idents_only):
    toks = query_tokens(query, idents_only)
    def order(ps):
        if not ps or not toks:
            return order_roundrobin(ps)
        rr = order_roundrobin(ps)
        pos = {id(p): i for i, p in enumerate(rr)}
        texts = {id(p): page_text(p) for p in ps}
        return sorted(ps, key=lambda p: (-sum(1 for t in toks if t in texts[id(p)]), pos[id(p)]))
    return order

QUERY_ORDERINGS = {"query-idents": True, "query-tokens": False}   # name -> idents_only

# ---------------------------------------------------------------- excerpt rules

IDENT = re.compile(r"(/[A-Za-z0-9_.\-{}]+){2,}|[A-Za-z]+[A-Z][A-Za-z0-9]+|[A-Za-z0-9]+(?:_[A-Za-z0-9]+){1,}|`[^`]+`")

def ex_longest(p):        # production
    return max((l["snippet"] for l in p["listings"]), key=len, default="")

def ex_identifiers(p):    # the snippet with the most path / camelCase / snake_case / backtick tokens
    return max((l["snippet"] for l in p["listings"]), key=lambda s: (len(IDENT.findall(s)), len(s)), default="")

def ex_all(p):            # every distinct snippet and excerpt joined, cut at the harness cap
    seen, parts = set(), []
    for l in p["listings"]:
        for s in [l["snippet"], *l["excerpts"]]:
            s = s.strip()
            if s and s not in seen:
                seen.add(s); parts.append(s)
    return " … ".join(parts)[:SNIPPET_CAP]

EXCERPTS = {"longest (prod)": ex_longest, "identifiers": ex_identifiers, "all-joined": ex_all}

# ---------------------------------------------------------------- scoring

def gold_hit(url, gold):
    return any(url_was_observed(g, [url]) for g in gold)

def job_scores(row, gold, required, goal="", use_jev=False):
    bodies = bodies_for(row)
    queries = [h.get("query") or "" for h in row.get("search_hops") or []]
    out = {}
    orderings = dict(ORDERINGS)
    for qname in QUERY_ORDERINGS:
        orderings[qname] = None
    if use_jev:
        orderings["jev score"] = None
    for oname, order in orderings.items():
        for ename, excerpt in EXCERPTS.items():
            seen = tok = False
            for bi, body in enumerate(bodies):
                q = queries[bi] if bi < len(queries) else ""
                if order is not None:
                    o = order
                elif oname == "jev score":
                    o = make_order_jev(q, goal)
                else:
                    o = make_order_tokens(q, QUERY_ORDERINGS[oname])
                for p in o(pages(body))[:TOP]:
                    if gold_hit(p["url"], gold):
                        seen = True
                        text = excerpt(p)[:SNIPPET_CAP]
                        if any(t in text for t in required):
                            tok = True
            out[(oname, ename)] = (seen, tok)
    # the upstream ceiling: gold anywhere in any provider's list
    up = any(gold_hit(p["url"], gold) for body in bodies for p in pages(body))
    return out, up


def main(argv=None) -> int:
    global _jev_cache, _jev_cache_path, TOP
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("passes", nargs="+", type=Path)
    ap.add_argument("--catalog", type=Path, required=True)
    ap.add_argument("--md", type=Path)
    ap.add_argument("--arms", default=",".join(sorted(FANOUT_ARMS)))
    ap.add_argument("--jev", action="store_true", help="add a fifth ordering scored by TypeSafe Jev (needs a key)")
    ap.add_argument("--jev-cache", type=Path, default=Path("rescore-jev-cache.json"))
    ap.add_argument("--top", type=int, default=TOP, help="hits shown per search (the harness shows 8); the window the gold page must land in")
    a = ap.parse_args(argv)
    TOP = a.top
    if a.jev:
        if not jev_key():
            sys.exit("--jev needs JEV_API_KEY (or ~/.config/search-bench/typesafe.key)")
        _jev_cache_path = a.jev_cache
        if _jev_cache_path.is_file():
            _jev_cache = json.loads(_jev_cache_path.read_text())
    cat = {i["id"]: i for i in json.loads(a.catalog.read_text())["items"]}
    arms = set(a.arms.split(","))
    per = defaultdict(list); upstream = defaultdict(list)
    for path in a.passes:
        for row in json.loads(path.read_text())["results"]:
            arm = row["search_backend"]
            if arm not in arms or row.get("error") or row["id"] not in cat:
                continue
            item = cat[row["id"]]
            gold = item.get("docs") or []
            if not gold:
                continue
            sc, up = job_scores(row, gold, item.get("required") or [], goal=item.get("prompt") or item.get("ticket") or "", use_jev=a.jev)
            per[arm].append(sc); upstream[arm].append(up)
    lines = []
    for arm in sorted(per):
        rows = per[arm]; n = len(rows)
        lines += [f"\n### {arm} (n = {n}; gold retrieved by some provider in {100*sum(upstream[arm])/n:.0f}% of jobs; window = top-{TOP})\n",
                  f"| ordering | excerpt | gold in top-{TOP} | required token in a top-{TOP} excerpt |", "|---|---|---|---|"]
        for oname in list(ORDERINGS) + list(QUERY_ORDERINGS) + (["jev score"] if a.jev else []):
            for ename in EXCERPTS:
                seen = sum(r[(oname, ename)][0] for r in rows); tok = sum(r[(oname, ename)][1] for r in rows)
                lines.append(f"| {oname} | {ename} | {100*seen/n:.0f}% | {100*tok/n:.0f}% |")
    table = "\n".join(lines)
    print(table)
    if a.md:
        a.md.write_text(table + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
