#!/usr/bin/env python3
"""The one command a headless Claude Code session may run: `web_search "<query>"`.

The driver (cc_run.py) writes a tiny shell script per job that sets CC_* and execs this file
with the harness's Python. The session sees only this command's stdout, which is the same shape
the harness's `web_search` tool returns: {"backend", "query", "hits": [{url, title, snippet}]}
plus "error" when the search failed. Every call is recorded as a hop file the driver turns into
the harness's search_hops and raw dumps, so the judge's grounding check and analyze.py work
unchanged.

Environment set by the generated script:
  CC_ROOT          the fork's root (for .env and the coding_search package)
  CC_ARM           backend name (a harness vendor or a Jevlin arm)
  CC_HOP_DIR       where hop files go (search-01.json, search-02.json, ...)
  CC_MAX_SEARCHES  budget; the (N+1)th call answers "search budget exhausted; answer now"
  CC_MAX_RESULTS   hits shown per search (8, like every vendor in the harness)
  CC_FIXTURE_PATH  tests only: with CC_ARM=jevlin_fixture, serve this merged-response fixture
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("CC_ROOT") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(ROOT / "scripts"))

from coding_search import search as harness_search  # noqa: E402
from coding_search.env import load_environment  # noqa: E402

BUDGET_EXHAUSTED = {"error": "search budget exhausted; answer now"}


def hop_files(hop_dir: Path) -> list[Path]:
    return sorted(hop_dir.glob("search-*.json"))


def emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def run_search(arm: str, query: str, max_results: int) -> tuple[list[dict], str | None, dict]:
    if arm == "jevlin_fixture":
        from coding_search import jevlin

        payload = json.loads(Path(os.environ["CC_FIXTURE_PATH"]).read_text())
        hits = jevlin.present_merged(payload, max_results=max_results)
        meta = {
            "endpoint": "fixture", "http_status": 200, "http_ms": 1,
            "request": {"method": "FIXTURE", "url": "fixture", "headers": {}, "params": None, "json": {"query": query}},
            "response": {"http_status": 200, "headers": {}, "body": payload},
            **harness_search._search_meta({}, hits),
            **jevlin.jevlin_meta(payload, hits, presentation="merged"),
        }
        return hits, None, meta
    backend = harness_search.get_backend(arm)
    try:
        hits = backend.search(query, max_results=max_results)
        error = None
    except Exception as exc:  # noqa: BLE001
        hits = []
        error = f"{type(exc).__name__}: {exc}"
    meta = dict(getattr(backend, "last_meta", None) or {})
    return hits, error, meta


def main(argv: list[str]) -> int:
    if len(argv) != 2 or not argv[1].strip():
        emit({"error": 'usage: web_search "your search query" (exactly one quoted argument)'})
        return 64
    query = argv[1].strip()
    arm = (os.environ.get("CC_ARM") or "").strip()
    hop_dir = Path(os.environ.get("CC_HOP_DIR") or "")
    max_searches = int(os.environ.get("CC_MAX_SEARCHES") or 5)
    max_results = int(os.environ.get("CC_MAX_RESULTS") or harness_search.DEFAULT_MAX_RESULTS)
    if not arm or not str(hop_dir):
        emit({"error": "web_search is not configured for this session"})
        return 70
    hop_dir.mkdir(parents=True, exist_ok=True)
    done = len(hop_files(hop_dir))
    if done >= max_searches:
        (hop_dir / "budget_exhausted").write_text(str(done) + "\n")
        emit(BUDGET_EXHAUSTED)
        return 0
    load_environment()
    started = time.perf_counter()
    hits, error, meta = run_search(arm, query, max_results)
    latency_ms = int((time.perf_counter() - started) * 1000)
    stats = harness_search.hit_text_stats(hits)
    hit_urls = [str(h.get("url") or "") for h in hits if h.get("url")] or list(meta.get("hit_urls") or [])
    seq = done + 1
    hop = {
        "query": query,
        "backend": arm,
        "hits": [{"url": str(h.get("url") or ""), "title": str(h.get("title") or ""), "snippet": str(h.get("snippet") or "")[:1200]} for h in hits],
        "error": error,
        "latency_ms": latency_ms,
        "n_hits": stats["n_hits"],
        "chars": stats["chars"],
        "tokens_est": stats["tokens_est"],
        "turn": 0,
        "seq": seq,
        "endpoint": str(meta.get("endpoint") or ""),
        "http_status": meta.get("http_status"),
        "http_ms": meta.get("http_ms"),
        "vendor_request_id": str(meta.get("vendor_request_id") or ""),
        "hit_urls": hit_urls,
        "empty": stats["n_hits"] == 0,
        "gold_token": None,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "request": meta.get("request") if isinstance(meta.get("request"), dict) else None,
        "response": meta.get("response") if isinstance(meta.get("response"), dict) else None,
        "vendor_meta": {k: v for k, v in meta.items() if k not in ("request", "response")},
    }
    (hop_dir / f"search-{seq:02d}.json").write_text(json.dumps(hop, ensure_ascii=False, default=str))
    shown: dict = {"backend": arm, "query": query, "hits": hop["hits"]}
    if error:
        shown["error"] = error
    emit(shown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
