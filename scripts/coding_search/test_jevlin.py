"""Offline contracts for the Jevlin arms: request shape, presentation, metadata, failures.

No network. The HTTP arms are exercised by patching the harness's vendor_call;
the CLI arm by patching subprocess.run. Fixtures are hand-built from the
router's response structs (fanout.Response, fanout.MergedResult,
provider.Candidate) and the client's searchEnvelope at the pinned commits.
"""

from __future__ import annotations

import json
import os
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

from coding_search import jevlin, judge, search

HERE = Path(__file__).resolve().parent
MERGED = json.loads((HERE / "testdata" / "jevlin" / "merged_response.json").read_text())
CLI_OK = json.loads((HERE / "testdata" / "jevlin" / "cli_envelope.json").read_text())

VERSIONS_URL = "https://servicenow.com/docs/r/api-reference/rest-apis/versions-api.html"
SHARED_URL = "https://servicenow.com/docs/r/api-reference/rest-apis/cdm-shared-libraries-api.html"
CDM_URL = "https://developer.servicenow.com/dev.do/reference/api/latest/rest/cdm-api"
THREAD_URL = "https://community.servicenow.com/cdm-snapshots-thread"

ENV = {
    "JEVLIN_API_KEY": "test-key",
    "JEVLIN_API_URL": "https://jevlin.test",
    "JEVLIN_PROVIDERS": "exa,brave,serpapi,tinyfish,you",
}


def _env(**extra: str):
    return patch.dict(os.environ, {**ENV, **extra}, clear=False)


class Registration(unittest.TestCase):
    def test_every_arm_is_search_only_and_fetch_forbidden(self):
        for name in jevlin.SEARCH_ONLY:
            with self.subTest(arm=name):
                self.assertIn(name, search.SEARCH_ONLY_BACKENDS)
                self.assertIn(name, search.FETCH_FORBIDDEN)
                self.assertNotIn(name, search.SEARCH_FETCH_BACKENDS)
                backend = search.get_backend(name)
                self.assertEqual(backend.name, name)
                with self.assertRaises(RuntimeError):
                    backend.fetch("https://example.com", objective="")

    def test_upstream_roster_is_untouched(self):
        # The upstream vendors still resolve, and nothing of theirs moved board.
        self.assertEqual(search.resolve_backend("exa"), "exa_auto")
        self.assertIn("perplexity_low", search.SEARCH_ONLY_BACKENDS)
        self.assertIn("exa_deep", search.SEARCH_FETCH_BACKENDS)
        self.assertNotIn("exa_deep", search.FETCH_FORBIDDEN)


class RequestShape(unittest.TestCase):
    def _capture(self, arm: str, *, max_results: int = 8, **env: str):
        with _env(**env), patch.object(search, "vendor_call", return_value=(MERGED, {})) as call:
            search.get_backend(arm).search("How do I publish a CDM snapshot?", max_results=max_results)
        self.assertEqual(call.call_count, 1)
        args, kwargs = call.call_args
        return args, kwargs

    def test_tier_arms_send_tier_and_no_providers(self):
        for arm, tier in (("jevlin_fast", "fast"), ("jevlin_balanced", "balanced"), ("jevlin_max", "max")):
            with self.subTest(arm=arm):
                args, kwargs = self._capture(arm)
                self.assertEqual(args[1:3], ("POST", "https://jevlin.test/v1/search"))
                self.assertEqual(kwargs["json_body"], {
                    "query": "How do I publish a CDM snapshot?",
                    "max_results": 8,
                    "view": "merged",
                    "tier": tier,
                })
                self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-key")
                self.assertEqual(kwargs["timeout"], jevlin.DEFAULT_TIMEOUT_S)

    def test_fanout_arms_pin_the_provider_set(self):
        for arm in ("jevlin_fanout_raw", "jevlin_fanout_merged"):
            with self.subTest(arm=arm):
                _, kwargs = self._capture(arm, max_results=5)
                self.assertEqual(kwargs["json_body"], {
                    "query": "How do I publish a CDM snapshot?",
                    "max_results": 5,
                    "view": "merged",
                    "tier": "balanced",
                    "providers": ["exa", "brave", "serpapi", "tinyfish", "you"],
                })

    def test_fanout_arms_refuse_without_a_pinned_set(self):
        for arm in ("jevlin_fanout_raw", "jevlin_fanout_merged"):
            with self.subTest(arm=arm), _env(JEVLIN_PROVIDERS=""), patch.object(search, "vendor_call") as call:
                with self.assertRaises(RuntimeError):
                    search.get_backend(arm).search("q")
                self.assertEqual(call.call_count, 0)

    def test_single_arm_names_one_provider_and_no_tier(self):
        _, kwargs = self._capture("jevlin_single")
        self.assertEqual(kwargs["json_body"]["providers"], ["exa"])
        self.assertNotIn("tier", kwargs["json_body"])
        _, kwargs = self._capture("jevlin_single", JEVLIN_SINGLE_PROVIDER="brave")
        self.assertEqual(kwargs["json_body"]["providers"], ["brave"])

    def test_missing_key_is_refused_before_any_call(self):
        with patch.dict(os.environ, {"JEVLIN_API_KEY": ""}), patch.object(search, "vendor_call") as call:
            with self.assertRaises(RuntimeError):
                search.get_backend("jevlin_balanced").search("q")
            self.assertEqual(call.call_count, 0)

    def test_timeout_and_url_env_overrides(self):
        _, kwargs = self._capture("jevlin_balanced", JEVLIN_TIMEOUT_S="12", JEVLIN_API_URL="https://beta.jevlin.test/")
        self.assertEqual(kwargs["timeout"], 12.0)
        self.assertEqual(self._capture("jevlin_balanced", JEVLIN_API_URL="https://beta.jevlin.test/")[0][2],
                         "https://beta.jevlin.test/v1/search")


class Presentation(unittest.TestCase):
    def test_merged_keeps_jevlin_order_fields_and_cap(self):
        hits = jevlin.present_merged(MERGED, max_results=8)
        self.assertEqual([h["url"] for h in hits], [VERSIONS_URL, SHARED_URL, CDM_URL, THREAD_URL])
        for hit in hits:
            self.assertEqual(set(hit), {"url", "title", "snippet"})
        self.assertTrue(hits[0]["snippet"].startswith("The Versions API provides endpoints"))
        self.assertEqual([h["url"] for h in jevlin.present_merged(MERGED, max_results=2)], [VERSIONS_URL, SHARED_URL])

    def test_merged_falls_back_to_excerpts_and_cuts_long_snippets(self):
        payload = {"merged": [{"url": "https://a.test/x", "excerpts": ["one", "two"]},
                              {"url": "https://a.test/y", "excerpt": "z" * 5000}]}
        hits = jevlin.present_merged(payload, max_results=8)
        self.assertEqual(hits[0]["snippet"], "one\ntwo")
        self.assertEqual(len(hits[1]["snippet"]), jevlin.SNIPPET_CHARS)

    def test_raw_round_robins_ok_candidates_and_dedups_exact_urls(self):
        hits = jevlin.present_raw(MERGED, max_results=8)
        # exa#1, brave#1, exa#2, (brave#2 duplicate), exa#3, (brave#3 duplicate)
        self.assertEqual([h["url"] for h in hits], [VERSIONS_URL, CDM_URL, SHARED_URL, THREAD_URL])
        # the first provider to show a page supplies its snippet: exa's shorter one, not brave's
        self.assertTrue(hits[0]["snippet"].startswith("PUT /api/sn_cdm/versions/publish"))
        self.assertEqual([h["url"] for h in jevlin.present_raw(MERGED, max_results=2)], [VERSIONS_URL, CDM_URL])

    def test_raw_ignores_pending_failed_and_skipped_arms(self):
        payload = {"candidates": [
            {"provider": "a", "status": "pending", "citations": [{"url": "https://p.test"}]},
            {"provider": "b", "status": "timeout", "citations": [{"url": "https://t.test"}]},
            {"provider": "c", "status": "ok", "citations": [{"url": "https://ok.test", "title": "T"}]},
        ]}
        self.assertEqual(jevlin.present_raw(payload, max_results=8), [{"url": "https://ok.test", "title": "T", "snippet": ""}])

    def test_backends_use_their_presentation(self):
        with _env(), patch.object(search, "vendor_call", return_value=(MERGED, {})):
            merged_hits = search.get_backend("jevlin_fanout_merged").search("q", max_results=8)
            raw_hits = search.get_backend("jevlin_fanout_raw").search("q", max_results=8)
        self.assertEqual([h["url"] for h in merged_hits][1], SHARED_URL)
        self.assertEqual([h["url"] for h in raw_hits][1], CDM_URL)


class Metadata(unittest.TestCase):
    def test_meta_records_what_the_harness_cannot(self):
        with _env(), patch.object(search, "vendor_call", return_value=(MERGED, {"http_status": 200, "http_ms": 12100})):
            backend = search.get_backend("jevlin_fanout_merged")
            hits = backend.search("q", max_results=8)
        meta = backend.last_meta
        self.assertEqual(meta["hit_urls"], [h["url"] for h in hits])
        self.assertEqual(meta["n_hits"], 4)
        self.assertEqual(meta["jevlin_request_id"], "req_fixture_0001")
        self.assertEqual(meta["jevlin_tier"], "balanced")
        self.assertEqual(meta["jevlin_providers"], ["exa", "brave", "serpapi", "tinyfish", "you"])
        self.assertTrue(meta["jevlin_requested"])
        self.assertEqual(meta["jevlin_cost_micros"], 10000)
        self.assertFalse(meta["jevlin_cache_hit"])
        self.assertEqual(meta["jevlin_pending"], 1)
        # exa, brave, serpapi (pending), tinyfish (timeout) were called; `you` was skipped
        self.assertEqual(meta["jevlin_upstream_calls"], 4)
        self.assertEqual(meta["jevlin_arms_ok"], 2)
        self.assertEqual(meta["jevlin_merged_total"], 4)
        self.assertEqual(meta["jevlin_hits_shown"], 4)
        self.assertEqual(meta["jevlin_router_version"], "none")
        skipped = [arm for arm in meta["jevlin_arms"] if arm["provider"] == "you"][0]
        self.assertEqual(skipped["reason"], "no_credential")
        self.assertEqual(meta["http_status"], 200)

    def test_no_usable_result_is_empty_hits_not_an_error(self):
        payload = {
            "request_id": "req_dead",
            "candidates": [{"provider": "exa", "status": "timeout"}, {"provider": "brave", "status": "error", "error": "upstream 500"}],
            "decision": {"tier": "balanced", "providers": ["exa", "brave"]},
            "usage": {"cost_micros": 7000, "cache_hit": False},
            "no_usable_result": True,
            "merged": [],
        }
        with _env(), patch.object(search, "vendor_call", return_value=(payload, {})):
            backend = search.get_backend("jevlin_balanced")
            self.assertEqual(backend.search("q"), [])
        self.assertTrue(backend.last_meta["jevlin_no_usable_result"])
        self.assertTrue(backend.last_meta["empty"])
        self.assertEqual(backend.last_meta["jevlin_upstream_calls"], 2)
        self.assertEqual(backend.last_meta["jevlin_cost_micros"], 7000)

    def test_http_errors_propagate_so_the_harness_records_them(self):
        err = requests.HTTPError("402 for https://jevlin.test/v1/search")
        setattr(err, "vendor_meta", {"http_status": 402, "error_body": "spend_ceiling"})
        with _env(), patch.object(search, "vendor_call", side_effect=err):
            with self.assertRaises(requests.HTTPError):
                search.get_backend("jevlin_fast").search("q")


class JudgeCompatibility(unittest.TestCase):
    def test_jevlin_canonical_urls_satisfy_the_grounding_check(self):
        observed = [h["url"] for h in jevlin.present_merged(MERGED, max_results=8)]
        # The model may copy a www / trailing-slash / query-string variant into `# source:`.
        for cited in (
            "https://www.servicenow.com/docs/r/api-reference/rest-apis/versions-api.html",
            "https://servicenow.com/docs/r/api-reference/rest-apis/versions-api.html/",
            "http://servicenow.com/docs/r/api-reference/rest-apis/versions-api.html?utm_source=x",
        ):
            with self.subTest(cited=cited):
                self.assertTrue(judge.url_was_observed(cited, observed))
        self.assertFalse(judge.url_was_observed("https://servicenow.com/docs/other.html", observed))


def _completed(stdout: str, *, code: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["jevlin", "search", "--stdin"], returncode=code, stdout=stdout, stderr=stderr)


class CliArm(unittest.TestCase):
    def test_runs_the_cli_and_shows_the_client_merge(self):
        with patch.dict(os.environ, {"JEVLIN_BIN": "/opt/jevlin/bin/jevlin", "JEVLIN_CLI_CONFIG": "/bench/jevlin.toml",
                                     "JEVLIN_CONFIG": "/home/operator/.jevlin/jevlin.toml"}), \
             patch.object(jevlin.subprocess, "run", return_value=_completed(json.dumps(CLI_OK))) as run:
            backend = search.get_backend("jevlin_cli")
            hits = backend.search("How do I publish a CDM snapshot?", max_results=2)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0], ["/opt/jevlin/bin/jevlin", "search", "-config", "/bench/jevlin.toml", "--stdin", "-no-flush"])
        # The operator's own JEVLIN_CONFIG must not reach the arm.
        self.assertEqual(run.call_args.kwargs["env"]["JEVLIN_CONFIG"], "/bench/jevlin.toml")
        self.assertEqual(json.loads(run.call_args.kwargs["input"]),
                         {"version": 1, "query": "How do I publish a CDM snapshot?", "max_results": 2})
        self.assertNotIn("tier", json.loads(run.call_args.kwargs["input"]))
        self.assertEqual([h["url"] for h in hits], [VERSIONS_URL, CDM_URL])
        self.assertEqual(set(hits[0]), {"url", "title", "snippet"})
        meta = backend.last_meta
        self.assertTrue(meta["jevlin_client_merge"])
        self.assertEqual(meta["jevlin_request_id"], "req_cli_0001")
        self.assertEqual(meta["jevlin_cost_micros"], 8000)
        self.assertEqual(meta["jevlin_upstream_calls"], 2)
        self.assertEqual(meta["jevlin_merged_total"], 3)
        self.assertEqual(meta["jevlin_hits_shown"], 2)
        self.assertEqual(meta["http_status"], 200)
        self.assertEqual(meta["request"]["json"]["query"], "How do I publish a CDM snapshot?")

    def test_ok_false_raises_with_the_client_code(self):
        envelope = {"ok": False, "http_status": 429, "retryable": True, "retry_after_ms": 5000,
                    "error": {"code": "rate_limited", "message": "slow down"}}
        with patch.object(jevlin.subprocess, "run", return_value=_completed(json.dumps(envelope), code=3)):
            backend = search.get_backend("jevlin_cli")
            with self.assertRaisesRegex(RuntimeError, "rate_limited"):
                backend.search("q")
        self.assertEqual(backend.last_meta["http_status"], 429)
        self.assertIn("rate_limited", backend.last_meta["error_body"])

    def test_total_failure_behind_ok_true_is_empty_and_flagged(self):
        envelope = {"ok": True, "http_status": 200, "result": {"request_id": "r", "chosen": -1, "merged": [],
                    "candidates": [{"provider": "exa", "status": "timeout"}, {"provider": "brave", "status": "error"}]}}
        with patch.object(jevlin.subprocess, "run", return_value=_completed(json.dumps(envelope))):
            backend = search.get_backend("jevlin_cli")
            self.assertEqual(backend.search("q"), [])
        self.assertTrue(backend.last_meta["jevlin_no_usable_result"])
        self.assertTrue(backend.last_meta["empty"])

    def test_missing_binary_and_timeout_are_errors(self):
        with patch.object(jevlin.subprocess, "run", side_effect=FileNotFoundError("jevlin")):
            with self.assertRaisesRegex(RuntimeError, "FileNotFoundError"):
                search.get_backend("jevlin_cli").search("q")
        with patch.object(jevlin.subprocess, "run", side_effect=subprocess.TimeoutExpired(cmd="jevlin", timeout=45)):
            backend = search.get_backend("jevlin_cli")
            with self.assertRaisesRegex(RuntimeError, "TimeoutExpired"):
                backend.search("q")
        self.assertIn("TimeoutExpired", backend.last_meta["error"])

    def test_garbage_stdout_is_an_error_not_a_crash(self):
        with patch.object(jevlin.subprocess, "run", return_value=_completed("not json", code=1, stderr="boom")):
            with self.assertRaises(RuntimeError):
                search.get_backend("jevlin_cli").search("q")


if __name__ == "__main__":
    unittest.main()
