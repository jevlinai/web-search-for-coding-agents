"""The whole harness loop on a Jevlin arm, with a scripted model and a fixture router.

No network, no paid call. A fake Responses client issues one web_search and
then returns a patched file; the Jevlin arm answers from the fixture; the
harness's own judge decides. This proves the plumbing end to end: request
shape, hit shape, hop recording, URL grounding, and the negative case where
the retrieved results do not contain the cited page.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from coding_search import agent, eval as harness_eval, judge, search
from coding_search.tasks import CodingTask

HERE = Path(__file__).resolve().parent
MERGED = json.loads((HERE / "testdata" / "jevlin" / "merged_response.json").read_text())

GOLD = '''import os
import httpx

BASE = os.environ.get("SN_BASE", "http://127.0.0.1:1950")


def run() -> dict:
    response = httpx.put(
        # source: https://www.servicenow.com/docs/r/api-reference/rest-apis/versions-api.html
        f"{BASE}/api/sn_cdm/versions/publish",
        json={"snapshot_sys_id": "abc"},
    )
    response.raise_for_status()
    return response.json()
'''

TASK = CodingTask(
    id="fixture_sn_publish",
    repo="servicenow",
    outfile="file.py",
    prompt="Patch run() so the snapshot version is published. Current file:\n```python\n...\n```",
    required=["sn_cdm/versions/publish"],
    query_forbidden=["sn_cdm/versions/publish"],
    docs=["https://www.servicenow.com/docs/r/api-reference/rest-apis/versions-api.html"],
)


class FakeResponses:
    """Scripted gpt-5.6-sol: one search, then the file."""

    def __init__(self, query: str, answer: str):
        self.query = query
        self.answer = answer
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        usage = SimpleNamespace(input_tokens=1000, output_tokens=50)
        if len(self.calls) == 1:
            call = SimpleNamespace(type="function_call", id="fc_1", call_id="call_1", name="web_search",
                                   arguments=json.dumps({"query": self.query}), status="completed")
            return SimpleNamespace(output=[call], output_text="", usage=usage)
        message = SimpleNamespace(type="message", id="msg_1", call_id=None, name=None, arguments=None, status="completed")
        return SimpleNamespace(output=[message], output_text=f"```python\n{self.answer}```", usage=usage)


class FakeClient:
    def __init__(self, query: str, answer: str):
        self.responses = FakeResponses(query, answer)


def _run(backend_name: str, payload: dict, *, answer: str = GOLD) -> tuple[agent.AgentRun, judge.JudgeScore, FakeClient]:
    client = FakeClient("servicenow cdm publish snapshot version api", answer)
    env = {"JEVLIN_API_KEY": "test-key", "JEVLIN_PROVIDERS": "exa,brave,serpapi,tinyfish,you"}
    with patch.dict(os.environ, env), patch.object(search, "vendor_call", return_value=(payload, {"http_status": 200, "http_ms": 900})):
        run = agent.run_agent(
            TASK.prompt,
            client=client,
            backend_name=backend_name,
            model="fake-model",
            max_turns=6,
            max_searches=5,
            max_fetches=0,
            system=harness_eval.PATCH_SYSTEM_SEARCH_ONLY,
            allow_search=True,
            allow_fetch=False,
            leak_needles=TASK.leak_needles(),
        )
    source = judge.extract_code(run.answer, TASK.outfile)
    score = judge.score_needles(source, TASK, observed_urls=judge.hop_urls(run), retrieved_text=judge.hop_retrieved_text(run))
    return run, score, client


class EndToEnd(unittest.TestCase):
    def test_merged_arm_grounds_the_citation(self):
        for arm in ("jevlin_fanout_merged", "jevlin_balanced", "jevlin_fanout_raw", "jevlin_single"):
            with self.subTest(arm=arm):
                run, score, client = _run(arm, MERGED)
                self.assertTrue(score.passed, score.output)
                self.assertEqual(run.backend, arm)
                self.assertEqual(len(run.searches), 1)
                hop = run.searches[0]
                self.assertEqual(hop.hit_urls[0], "https://servicenow.com/docs/r/api-reference/rest-apis/versions-api.html")
                self.assertIsNone(hop.error)
                self.assertEqual(hop.http_status, 200)
                self.assertEqual(hop.n_hits, 4)
                # What the model was shown: backend name, query, and url/title/snippet hits only.
                tool_output = json.loads(client.responses.calls[1]["input"][-1]["output"])
                self.assertEqual(set(tool_output), {"backend", "query", "hits"})
                self.assertEqual(set(tool_output["hits"][0]), {"url", "title", "snippet"})
                self.assertEqual(run.llm_prompt_tokens, 2000)

    def test_ungrounded_when_the_page_never_came_back(self):
        other = json.loads(json.dumps(MERGED))
        for page in other["merged"]:
            page["url"] = page["url"].replace("versions-api", "unrelated-api")
            page["excerpt"] = "nothing about publishing"
            page["title"] = "Other"
        for cand in other["candidates"]:
            for cit in cand.get("citations") or []:
                cit["url"] = cit["url"].replace("versions-api", "unrelated-api")
                cit["snippet"] = "nothing about publishing"
        run, score, _ = _run("jevlin_fanout_merged", other)
        self.assertFalse(score.passed)
        self.assertTrue(any(note.startswith("ungrounded:") for note in score.notes), score.notes)

    def test_total_provider_failure_is_recorded_as_an_empty_hop(self):
        dead = {"request_id": "r", "candidates": [{"provider": "exa", "status": "timeout"}],
                "decision": {"tier": "balanced", "providers": ["exa"]}, "usage": {"cost_micros": 1000},
                "no_usable_result": True, "merged": []}
        run, score, _ = _run("jevlin_balanced", dead)
        self.assertFalse(score.passed)
        self.assertTrue(run.searches[0].empty)
        self.assertIsNone(run.searches[0].error)

    def test_leak_flag_still_fires_on_jevlin_arms(self):
        client = FakeClient("sn_cdm/versions/publish endpoint", GOLD)
        env = {"JEVLIN_API_KEY": "test-key"}
        with patch.dict(os.environ, env), patch.object(search, "vendor_call", return_value=(MERGED, {})):
            run = agent.run_agent(TASK.prompt, client=client, backend_name="jevlin_balanced", model="fake",
                                 max_turns=4, max_searches=5, max_fetches=0,
                                 system=harness_eval.PATCH_SYSTEM_SEARCH_ONLY, allow_search=True,
                                 allow_fetch=False, leak_needles=TASK.leak_needles())
        self.assertEqual(run.searches[0].gold_token, "sn_cdm/versions/publish")


if __name__ == "__main__":
    unittest.main()
