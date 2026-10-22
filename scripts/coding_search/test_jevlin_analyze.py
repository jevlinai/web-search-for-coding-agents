"""run_arm → raw dumps → analyze.py, offline.

Drives the harness's own run_arm (not run_agent) on a Jevlin arm with the
scripted model and the fixture router, so the raw hop files are written by the
harness exactly as a real run writes them. Then analyze.py must find the
gateway body in those files and recover the Jevlin measures: upstream calls,
observed cost, cache hits, the gold-URL check and the single-provider oracle.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from coding_search import eval as harness_eval, jevlin, search
from coding_search.tasks import CodingTask
from coding_search.test_jevlin_e2e import GOLD, MERGED, FakeClient

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "jevlin_bench"))
import analyze  # noqa: E402


class AnalyzeFromRawDumps(unittest.TestCase):
    def test_measures_recovered_from_the_harness_dump(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "tasks" / "fixture_sn_publish"
            (task_dir / "gold").mkdir(parents=True)
            (task_dir / "starter").mkdir(parents=True)
            (task_dir / "gold" / "file.py").write_text(GOLD)
            (task_dir / "starter" / "file.py").write_text("import httpx\n")
            task = CodingTask(
                id="fixture_sn_publish", repo="servicenow", outfile="file.py",
                prompt="Patch run().", required=["sn_cdm/versions/publish"], query_forbidden=["sn_cdm/versions/publish"],
                docs=["https://www.servicenow.com/docs/r/api-reference/rest-apis/versions-api.html"], task_dir=task_dir,
            )
            vendor_meta = {
                "endpoint": "https://jevlin.test/v1/search", "method": "POST", "http_status": 200, "http_ms": 900,
                "response_bytes": 1234, "vendor_request_id": "",
                "request": {"method": "POST", "url": "https://jevlin.test/v1/search", "headers": {"Authorization": "***REDACTED***"},
                            "params": None, "json": {"query": "q", "view": "merged"}},
                "response": {"http_status": 200, "headers": {}, "body": MERGED},
            }
            env = {"JEVLIN_API_KEY": "test-key", "JEVLIN_PROVIDERS": "exa,brave,serpapi,tinyfish,you"}
            rows = []
            for arm in ("jevlin_fanout_merged", "jevlin_fanout_raw"):
                with patch.dict(os.environ, env), patch.object(search, "vendor_call", return_value=(MERGED, vendor_meta)):
                    rows.append(harness_eval.run_arm(
                        task, client=FakeClient("servicenow cdm publish snapshot version api", GOLD),
                        backend=arm, allow_fetch=False, model="fake", max_turns=6, max_searches=5, max_fetches=0,
                        raw_dir=root / "raw", dataset="public", split="search-only", run_stamp="2026-10-15T00-00-00Z",
                    ))
            for row in rows:
                self.assertTrue(row["passed"], row["score"])
                hop_file = Path(row["raw_dir"]) / "hops" / "search-01.response.json"
                self.assertTrue(hop_file.is_file())
                self.assertEqual(json.loads(hop_file.read_text())["body"]["request_id"], "req_fixture_0001")
            summary_path = root / "pass-1.json"
            summary_path.write_text(json.dumps({"max_results": 8, "results": rows}))
            catalog_path = root / "catalog.json"
            catalog_path.write_text(json.dumps({"items": [{"id": task.id, "docs": task.docs}]}))
            report_path = root / "report.json"
            with patch("sys.stdout"):
                analyze.main(["--out", str(summary_path), "--catalog", str(catalog_path), "--json", str(report_path)])
            report = json.loads(report_path.read_text())
            for arm in ("jevlin_fanout_merged", "jevlin_fanout_raw"):
                stats = report[arm]
                self.assertEqual(stats["rows"], 1)
                self.assertEqual(stats["completion_mean"], 1.0)
                self.assertEqual(stats["searches_per_task"], 1.0)
                # exa, brave, serpapi (pending), tinyfish (timeout); `you` was skipped
                self.assertEqual(stats["upstream_per_task"], 4.0)
                self.assertEqual(stats["cost_usd_per_task"], 0.01)
                self.assertEqual(stats["cost_kind"], "observed")
                self.assertEqual(stats["cache_hit_searches"], 0)
                self.assertEqual(stats["pending_arms"], 1)
                self.assertEqual(stats["gold_url_retrieved"], 1.0)
                # exa alone (and brave alone) returned the gold page: the oracle bound is met
                self.assertEqual(stats["oracle_single_provider_retrieval"], 1.0)
                self.assertEqual(stats["providers_seen"], ["brave", "exa", "serpapi", "tinyfish", "you"])
                self.assertEqual(stats["hop_errors"], 0)


class AnalyzeSplitArm(unittest.TestCase):
    def test_split_numbers_are_recomputed_from_the_hits_each_search_showed(self):
        def hit(i, compact):
            return {"url": f"https://p.test/{i}", "title": f"Page {i}", "snippet": "s" * 20, **({"compact": True} if compact else {})}

        def row(arm, searches):
            return {"id": "t1", "search_backend": arm, "backend": f"{arm}+search", "passed": True, "search_count": len(searches),
                    "metrics": {}, "search_hops": [{"hits": hits, "hit_urls": [h["url"] for h in hits]} for hits in searches]}

        full = [hit(i, i >= 10) for i in range(40)]
        short = [hit(i, False) for i in range(10)]  # a short list, or an answer from the cache of a plain search
        rows = [row("jevlin_balanced_split", [full, short]), row("jevlin_balanced", [short])]
        with tempfile.TemporaryDirectory() as tmp:
            summary = Path(tmp) / "pass-1.json"
            summary.write_text(json.dumps({"max_results": 8, "results": rows}))
            report_path = Path(tmp) / "report.json"
            with patch("sys.stdout"):
                analyze.main(["--out", str(summary), "--json", str(report_path)])
            report = json.loads(report_path.read_text())
        split, plain = report["jevlin_balanced_split"], report["jevlin_balanced"]
        self.assertEqual((split["split_entries_per_search"], split["split_compact_per_search"]), (25.0, 15.0))
        self.assertEqual(split["split_share_searches_with_compact"], 0.5)
        expected = [jevlin.split_stats(jevlin.shown_hits(h))["jevlin_split_chars_shown"] for h in (full, short)]
        self.assertEqual(split["split_chars_per_search"], sum(expected) / 2)
        for key in ("split_entries_per_search", "split_compact_per_search", "split_chars_per_search", "split_share_searches_with_compact"):
            self.assertIsNone(plain[key], key)


if __name__ == "__main__":
    unittest.main()
