"""Offline contracts for the split arm, `jevlin_balanced_split`.

The arm sends `jevlin_balanced`'s request plus `excerpt_mode:"split"` and shows the model the router's
whole merged list: the long entries as the harness's hit objects, the compact tail as one line each. This
file covers its request, its presentation, the metadata it records, the wrapper that renders it, the router
probe that gates it, and that every other arm is untouched.

No network. Every test runs under `no_network()`, which makes any request fail loudly (a past test reached
the router and made a real, billed call); a test that needs the router's answer stages it by patching the
request function inside that guard. Fixtures are synthetic: the router's split response shape is assumed
from its description (merged[] entries with `excerpt`, the tail marked `compact: true`), not captured.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import requests

from coding_search import eval as harness_eval
from coding_search import jevlin, search

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "jevlin_bench"))
import cc_run  # noqa: E402
import cc_search  # noqa: E402
import split_preflight  # noqa: E402

MERGED = json.loads((HERE / "testdata" / "jevlin" / "merged_response.json").read_text())
QUERY = "How do I publish a CDM snapshot?"
ENV = {"JEVLIN_API_KEY": "test-key", "JEVLIN_API_URL": "https://jevlin.test", "JEVLIN_PROVIDERS": "exa,brave,serpapi,tinyfish,you"}

# What the router answers the probe with: flat {"error", "code"}, the code is the discriminator.
UNKNOWN_FIELD = {"error": 'json: unknown field "excerpt_mode"', "code": "malformed_json"}
VALIDATED = {"error": "excerpt_mode split requires view merged and tier balanced or max", "code": "invalid_request"}


def no_network() -> contextlib.ExitStack:
    """Make any request fail loudly, under every name the harness can reach one by."""
    stack = contextlib.ExitStack()
    refusal = AssertionError("the test suite must not reach the network")
    for target in ("coding_search.search.vendor_call", "coding_search.search.vendor_request", "requests.sessions.Session.request"):
        stack.enter_context(patch(target, side_effect=refusal))
    return stack


def split_payload(*, long: int = 10, compact: int = 30) -> dict:
    """A synthetic split-view response: `long` entries with a long excerpt, then `compact` one-liners."""
    merged = [{"url": f"https://long.test/{i}", "title": f"Long page {i}", "excerpt": f"Long excerpt {i}. " + "x" * 300} for i in range(long)]
    merged += [{"url": f"https://compact.test/{i}", "title": f"Compact page {i}", "excerpt": f"short text {i}", "compact": True} for i in range(compact)]
    return {**MERGED, "merged": merged}


def _env(**extra: str):
    return patch.dict(os.environ, {**ENV, **extra}, clear=False)


class Guarded(unittest.TestCase):
    def setUp(self):
        guard = no_network()
        guard.__enter__()
        self.addCleanup(guard.__exit__, None, None, None)


class ArmAndRequest(Guarded):
    def _capture(self, arm: str, payload=None, *, max_results: int = 8):
        payload = split_payload() if payload is None else payload
        with _env(), patch.object(search, "vendor_call", return_value=(payload, {})) as call:
            hits = search.get_backend(arm).search(QUERY, max_results=max_results)
        self.assertEqual(call.call_count, 1)
        return hits, call.call_args

    def test_the_network_guard_refuses_a_request_the_test_did_not_stage(self):
        with _env(), self.assertRaisesRegex(AssertionError, "must not reach the network"):
            search.get_backend("jevlin_balanced_split").search(QUERY)

    def test_the_arm_is_a_search_only_backend_that_only_the_claude_code_driver_runs(self):
        backend = search.get_backend("jevlin_balanced_split")
        self.assertEqual((backend.name, backend.tier, backend.presentation, backend.excerpt_mode), ("jevlin_balanced_split", "balanced", "split", "split"))
        self.assertTrue(jevlin.is_split_arm("jevlin_balanced_split"))
        for other in ("jevlin_balanced", "jevlin_fast", "jevlin_max", "jevlin_cli", "perplexity_low", "nope"):
            self.assertFalse(jevlin.is_split_arm(other), other)
        with self.assertRaises(RuntimeError):
            backend.fetch("https://example.com", objective="")
        # kept off the API-agent roster: not in the board lists, not in `all`, refused by parse_arms
        self.assertEqual(jevlin.CC_ONLY, ("jevlin_balanced_split",))
        self.assertNotIn("jevlin_balanced_split", jevlin.SEARCH_ONLY)
        self.assertNotIn("jevlin_balanced_split", search.SEARCH_ONLY_BACKENDS)
        self.assertNotIn("jevlin_balanced_split", harness_eval.parse_arms("all", split="search-only"))
        with self.assertRaises(SystemExit) as refused:
            harness_eval.parse_arms("jevlin_balanced,jevlin_balanced_split")
        self.assertIn("cc_run.py", str(refused.exception))
        self.assertEqual(harness_eval.parse_arms("jevlin_balanced"), ("jevlin_balanced",))

    def test_request_is_the_balanced_body_plus_excerpt_mode_and_nothing_else(self):
        _, split = self._capture("jevlin_balanced_split")
        _, balanced = self._capture("jevlin_balanced")
        self.assertEqual(split.kwargs["json_body"], {**balanced.kwargs["json_body"], "excerpt_mode": "split"})
        self.assertEqual(split.kwargs["json_body"], {"query": QUERY, "max_results": 8, "view": "merged", "tier": "balanced", "excerpt_mode": "split"})
        self.assertEqual(split.args[1:], balanced.args[1:])  # same method and URL
        self.assertEqual({k: v for k, v in split.kwargs.items() if k != "json_body"}, {k: v for k, v in balanced.kwargs.items() if k != "json_body"})

    def test_no_other_backend_sends_excerpt_mode(self):
        for name in jevlin.SEARCH_ONLY:
            if name == "jevlin_cli":
                continue
            with self.subTest(arm=name), _env():
                self.assertNotIn("excerpt_mode", jevlin.BACKENDS[name]().request_body(QUERY, 8))
        with _env():
            self.assertEqual(jevlin.BACKENDS["jevlin_balanced"]().request_body(QUERY, 8),
                             {"query": QUERY, "max_results": 8, "view": "merged", "tier": "balanced"})

    def test_the_whole_merged_list_is_shown_in_router_order_with_no_cut(self):
        payload = split_payload()
        for max_results in (8, 3, 1):
            with self.subTest(max_results=max_results):
                hits, call = self._capture("jevlin_balanced_split", payload, max_results=max_results)
                self.assertEqual([h["url"] for h in hits], [m["url"] for m in payload["merged"]])
                self.assertEqual(call.kwargs["json_body"]["max_results"], max_results)  # still sent to the router
        self.assertEqual(len(hits), 40)
        for hit in hits[:10]:
            self.assertEqual(set(hit), {"url", "title", "snippet"})
        for hit in hits[10:]:
            self.assertEqual(set(hit), {"url", "title", "snippet", "compact"})
            self.assertIs(hit["compact"], True)
        balanced, _ = self._capture("jevlin_balanced", payload, max_results=10)
        self.assertEqual(hits[:10], balanced, "the long entries are what jevlin_balanced shows for the same pages")

    def test_compact_entries_are_one_line_each_and_long_entries_stay_objects(self):
        hits, _ = self._capture("jevlin_balanced_split")
        shown = jevlin.shown_hits(hits)
        self.assertEqual(shown[:10], hits[:10])
        self.assertTrue(all(isinstance(item, dict) for item in shown[:10]))
        self.assertTrue(all(isinstance(item, str) and "\n" not in item for item in shown[10:]))
        self.assertEqual(shown[10], "Compact page 0 | https://compact.test/0 | short text 0")
        self.assertEqual([line.split(" | ")[1] for line in shown[10:]], [f"https://compact.test/{i}" for i in range(30)])

    def test_a_compact_line_collapses_whitespace_and_leaves_out_empty_parts(self):
        hit = {"url": "https://a.test", "title": " A \n title ", "snippet": "two\nlines   and\ttabs", "compact": True}
        self.assertEqual(jevlin.compact_line(hit), "A title | https://a.test | two lines and tabs")
        self.assertEqual(jevlin.compact_line({"url": "https://a.test", "title": "", "snippet": "", "compact": True}), "https://a.test")
        self.assertEqual(jevlin.compact_line({"url": "https://a.test", "title": "T", "snippet": "", "compact": True}), "T | https://a.test")

    def test_metadata_records_entries_shown_compact_and_characters(self):
        with _env(), patch.object(search, "vendor_call", return_value=(split_payload(), {"http_status": 200})):
            backend = search.get_backend("jevlin_balanced_split")
            hits = backend.search(QUERY, max_results=8)
        meta = backend.last_meta
        shown = jevlin.shown_hits(hits)
        self.assertEqual((meta["jevlin_split_entries_shown"], meta["jevlin_split_compact_shown"]), (40, 30))
        self.assertEqual(meta["jevlin_split_chars_shown"], len(json.dumps(shown, ensure_ascii=False)))
        self.assertLess(meta["jevlin_split_chars_shown"], len(json.dumps(hits, ensure_ascii=False)))
        self.assertEqual((meta["jevlin_hits_shown"], meta["jevlin_merged_total"], meta["jevlin_presentation"], meta["n_hits"]), (40, 40, "split", 40))
        self.assertEqual(meta["jevlin_tier"], "balanced")

    def test_entries_the_harness_cannot_show_are_visible_as_the_gap_to_merged_total(self):
        payload = split_payload(long=2, compact=3)
        payload["merged"] += [{"url": "https://long.test/0", "title": "dup"}, {"title": "no url"}, "not a dict"]
        with _env(), patch.object(search, "vendor_call", return_value=(payload, {})):
            backend = search.get_backend("jevlin_balanced_split")
            backend.search(QUERY)
        self.assertEqual((backend.last_meta["jevlin_merged_total"], backend.last_meta["jevlin_split_entries_shown"]), (8, 5))

    def test_a_response_with_no_compact_entry_is_shown_as_it_came(self):
        hits, _ = self._capture("jevlin_balanced_split", split_payload(long=8, compact=0))
        self.assertEqual(len(hits), 8)
        self.assertEqual(jevlin.shown_hits(hits), hits)

    def test_other_arms_are_unchanged_by_the_split_work(self):
        payload = split_payload()
        for arm in ("jevlin_fast", "jevlin_balanced", "jevlin_max", "jevlin_fanout_merged"):
            with self.subTest(arm=arm), _env(), patch.object(search, "vendor_call", return_value=(payload, {})):
                backend = search.get_backend(arm)
                hits = backend.search(QUERY, max_results=8)
            self.assertEqual(len(hits), 8, "still the top-N cut")
            self.assertTrue(all(set(h) == {"url", "title", "snippet"} for h in hits))
            self.assertEqual(jevlin.shown_hits(hits), hits)
            self.assertFalse([key for key in backend.last_meta if "split" in key])
        self.assertEqual(sorted(jevlin.PRESENTATIONS), ["merged", "raw", "split"])
        self.assertEqual(len(jevlin.present_merged(payload, max_results=8)), 8)
        self.assertEqual(len(jevlin.present_raw(payload, max_results=8)), 4)  # the fixture's four unique candidate URLs


class Wrapper(Guarded):
    """The real cc_search.main and backend, with the router's answer staged."""

    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.hops = Path(self.tmp.name) / "hops"

    def _search(self, arm: str, payload: dict, *, max_results: int = 8):
        env = {**ENV, "CC_ARM": arm, "CC_HOP_DIR": str(self.hops), "CC_MAX_SEARCHES": "5", "CC_MAX_RESULTS": str(max_results)}
        with patch.dict(os.environ, env), patch.object(cc_search, "load_environment"), \
                patch.object(search, "vendor_call", return_value=(payload, {"endpoint": "https://jevlin.test/v1/search", "http_status": 200, "http_ms": 3})) as call, \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(cc_search.main(["web_search", QUERY]), 0)
        hop = json.loads(sorted(self.hops.glob("search-*.json"))[-1].read_text())
        return json.loads(out.getvalue()), hop, call.call_args.kwargs["json_body"]

    def test_the_model_reads_the_whole_list_long_entries_as_objects_and_the_tail_as_lines(self):
        shown, hop, body = self._search("jevlin_balanced_split", split_payload())
        self.assertEqual(body["excerpt_mode"], "split")
        self.assertEqual(set(shown), {"backend", "query", "hits"}, "the tool output keeps its shape")
        hits = shown["hits"]
        self.assertEqual(len(hits), 40)
        self.assertTrue(all(isinstance(h, dict) and set(h) == {"url", "title", "snippet"} for h in hits[:10]))
        self.assertTrue(all(isinstance(h, str) and "\n" not in h for h in hits[10:]))
        self.assertEqual(hits[10], "Compact page 0 | https://compact.test/0 | short text 0")
        self.assertEqual([h["url"] for h in hits[:10]], [f"https://long.test/{i}" for i in range(10)])
        self.assertEqual([h.split(" | ")[1] for h in hits[10:]], [f"https://compact.test/{i}" for i in range(30)])

    def test_the_cut_that_applies_to_every_other_arm_does_not_apply_to_the_split_arm(self):
        shown, _, body = self._search("jevlin_balanced_split", split_payload(), max_results=3)
        self.assertEqual((len(shown["hits"]), body["max_results"]), (40, 3))
        self.hops = Path(self.tmp.name) / "balanced"
        shown, _, _ = self._search("jevlin_balanced", split_payload(), max_results=3)
        self.assertEqual(len(shown["hits"]), 3)

    def test_the_hop_keeps_the_hits_and_the_list_as_shown(self):
        shown, hop, _ = self._search("jevlin_balanced_split", split_payload())
        self.assertEqual(hop["shown_hits"], shown["hits"])
        self.assertEqual(len(hop["hits"]), 40)
        self.assertEqual(sum(1 for h in hop["hits"] if h.get("compact")), 30)
        # the judge reads hop["hits"] and hop["hit_urls"]: all 40 pages count as observed
        self.assertEqual((hop["n_hits"], len(hop["hit_urls"])), (40, 40))
        self.assertIn("https://compact.test/29", hop["hit_urls"])

    def test_hop_metadata_matches_what_the_model_received(self):
        shown, hop, _ = self._search("jevlin_balanced_split", split_payload())
        meta = hop["vendor_meta"]
        self.assertEqual((meta["jevlin_split_entries_shown"], meta["jevlin_split_compact_shown"]), (40, 30))
        self.assertEqual(meta["jevlin_split_chars_shown"], len(json.dumps(shown["hits"], ensure_ascii=False)))
        self.assertEqual(sum(isinstance(h, str) for h in shown["hits"]), meta["jevlin_split_compact_shown"])

    def test_other_arms_are_shown_and_recorded_exactly_as_before(self):
        for arm in ("jevlin_balanced", "jevlin_fast"):
            with self.subTest(arm=arm):
                self.hops = Path(self.tmp.name) / arm
                shown, hop, body = self._search(arm, split_payload(), max_results=8)
                self.assertEqual(len(shown["hits"]), 8)
                self.assertTrue(all(isinstance(h, dict) and set(h) == {"url", "title", "snippet"} for h in shown["hits"]))
                self.assertNotIn("excerpt_mode", body)
                self.assertNotIn("shown_hits", hop)
                self.assertTrue(all("compact" not in h for h in hop["hits"]))
                self.assertFalse([key for key in hop["vendor_meta"] if "split" in key])


class FakeResponse:
    def __init__(self, status_code: int, payload=None, text: str = ""):
        self.status_code, self._payload, self.text = status_code, payload, text

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class Preflight(Guarded):
    def test_only_a_split_arm_needs_the_probe(self):
        def fail():
            raise AssertionError("probe must not run")
        self.assertIsNone(split_preflight.require_split_router(["jevlin_balanced", "jevlin_fixture", "exa_fast", "jevlin_cli"], probe_fn=fail))

    def test_outcome_one_a_refusal_naming_the_field_is_a_router_that_knows_it(self):
        accepted, detail = split_preflight.classify(400, VALIDATED)
        self.assertTrue(accepted, detail)
        self.assertTrue(split_preflight.classify(400, {"error": {"code": "invalid_request", "message": "Excerpt_Mode is not allowed on tier fast"}})[0])

    def test_outcome_two_malformed_json_is_a_router_without_the_field(self):
        accepted, detail = split_preflight.classify(400, UNKNOWN_FIELD)
        self.assertFalse(accepted)
        self.assertIn("malformed_json", detail)
        self.assertIn("unknown field", detail)

    def test_outcome_three_anything_that_is_not_a_readable_refusal_is_not_confirmation(self):
        accepted, detail = split_preflight.classify(200, {"merged": [], "request_id": "r"})
        self.assertFalse(accepted)
        self.assertIn("ignored the field", detail)
        self.assertIn("billed", detail)
        for status, payload in (
            (400, {"error": "query is required", "code": "invalid_request"}),  # a 400 that does not name excerpt_mode
            (401, {"error": "bad key", "code": "unauthorized"}),
            (403, {"error": "tier not allowed", "code": "forbidden"}),
            (429, {"error": "slow down", "code": "rate_limited"}),
            (500, "upstream exploded"),
            (None, None),
        ):
            with self.subTest(status=status):
                self.assertFalse(split_preflight.classify(status, payload)[0])

    def test_probe_sends_split_on_fast_once_and_reads_the_refusal(self):
        cases = [
            (FakeResponse(400, VALIDATED), True),
            (FakeResponse(400, UNKNOWN_FIELD), False),
            (FakeResponse(200, {"merged": []}), False),
            (FakeResponse(502, None, text="<html>bad gateway</html>"), False),
        ]
        for response, accepted in cases:
            with self.subTest(status=response.status_code), _env(), patch("requests.post", return_value=response) as post:
                self.assertEqual(split_preflight.probe()[0], accepted)
            self.assertEqual(post.call_count, 1)
            self.assertEqual(post.call_args.args, ("https://jevlin.test/v1/search",))
            self.assertEqual(post.call_args.kwargs["json"], {"query": split_preflight.PROBE_QUERY, "tier": "fast", "view": "merged",
                                                              "max_results": 1, "excerpt_mode": "split"})
            self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer test-key")

    def test_probe_without_a_key_or_an_answer_makes_no_claim(self):
        with patch.dict(os.environ, {"JEVLIN_API_KEY": ""}):
            accepted, detail = split_preflight.probe()  # the guard would raise if it tried to send
        self.assertFalse(accepted)
        self.assertIn("JEVLIN_API_KEY", detail)
        with _env(), patch("requests.post", side_effect=requests.ConnectionError("refused")):
            accepted, detail = split_preflight.probe()
        self.assertFalse(accepted)
        self.assertIn("no answer", detail)

    def test_the_arm_is_refused_unless_the_probe_confirms(self):
        arms = ["jevlin_balanced", "jevlin_balanced_split"]
        self.assertIsNone(split_preflight.require_split_router(arms, probe_fn=lambda: (True, "accepted")))
        message = split_preflight.require_split_router(arms, probe_fn=lambda: (False, "400 malformed_json: unknown field"))
        self.assertIn("excerpt_mode", message)
        self.assertIn("malformed_json", message)
        self.assertIn("before any provider is called", message)

    def test_the_driver_refuses_to_start_a_split_pass_unless_the_router_confirms(self):
        class Started(Exception):
            pass

        with tempfile.TemporaryDirectory() as tmp:
            argv = ["--arms", "jevlin_balanced,jevlin_balanced_split", "--out", str(Path(tmp) / "out" / "pass-1.json"), "--claude-bin", "/nonexistent/claude"]
            tasks = [SimpleNamespace(id="t-1")]
            out = io.StringIO()
            with patch.object(split_preflight, "probe", return_value=(False, "400 malformed_json")), patch.object(cc_run, "load_tasks", return_value=tasks), \
                    patch.object(cc_run, "run_job", side_effect=AssertionError("no session may start")), contextlib.redirect_stdout(out):
                self.assertEqual(cc_run.main(argv), 2)
            self.assertIn("excerpt_mode", out.getvalue())
            self.assertFalse((Path(tmp) / "out" / "pass-1.json").exists())
            # a router that confirms lets the first session start
            with patch.object(split_preflight, "probe", return_value=(True, "accepted")), patch.object(cc_run, "load_tasks", return_value=tasks), \
                    patch.object(cc_run, "claude_version", return_value="9.9.9"), patch.object(cc_run, "run_job", side_effect=Started), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(Started):
                    cc_run.main(argv)
            # a pass without the split arm never probes
            with patch.object(split_preflight, "probe", side_effect=AssertionError("no probe")), patch.object(cc_run, "load_tasks", return_value=tasks), \
                    patch.object(cc_run, "claude_version", return_value="9.9.9"), patch.object(cc_run, "run_job", side_effect=Started), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(Started):
                    cc_run.main(["--arms", "jevlin_balanced", "--out", str(Path(tmp) / "out" / "pass-1.json")])


if __name__ == "__main__":
    unittest.main()
