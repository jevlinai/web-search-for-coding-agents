"""The Claude client behind the harness loop, offline.

A fake Anthropic SDK records every request and returns scripted content. The
tests prove the translation both ways (system, messages, tools, tool_use →
function_call, tool_result grouping, thinking blocks re-attached), the usage
accounting, and that `agent.run_agent` runs unmodified end to end through it
on a Jevlin arm, with the judge passing on the fixture.
"""

from __future__ import annotations

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from coding_search import agent, claude_client, env, eval as harness_eval, judge, search
from coding_search.test_jevlin_e2e import GOLD, MERGED, TASK


def _block(**kwargs):
    return SimpleNamespace(**kwargs)


class FakeMessages:
    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        content = self.scripted.pop(0)
        return SimpleNamespace(
            content=content,
            usage=SimpleNamespace(input_tokens=900, output_tokens=40, cache_read_input_tokens=100, cache_creation_input_tokens=0),
            stop_reason="tool_use" if any(b.type == "tool_use" for b in content) else "end_turn",
            model="claude-test",
        )


class FakeAnthropic:
    def __init__(self, scripted):
        self.messages = FakeMessages(scripted)


SEARCH_TURN = [
    _block(type="thinking", thinking="I should search the docs.", signature="sig-1"),
    _block(type="tool_use", id="toolu_01", name="web_search", input={"query": "servicenow cdm publish snapshot version api"}),
]
ANSWER_TURN = [
    _block(type="thinking", thinking="Found it.", signature="sig-2"),
    _block(type="text", text=f"```python\n{GOLD}```"),
]


class Translation(unittest.TestCase):
    def test_first_turn_request_shape(self):
        fake = FakeAnthropic([SEARCH_TURN])
        client = claude_client.ClaudeResponsesClient(api_key="k", anthropic_client=fake)
        response = client.responses.create(
            model="claude-test",
            reasoning={"effort": "medium"},
            tools=[agent.WEB_SEARCH_TOOL],
            input=[{"role": "system", "content": "You edit scripts."}, {"role": "user", "content": "Patch run()."}],
            max_output_tokens=4000,
            store=False,
        )
        request = fake.messages.requests[0]
        self.assertEqual(request["model"], "claude-test")
        self.assertEqual(request["system"], "You edit scripts.")
        self.assertEqual(request["messages"], [{"role": "user", "content": "Patch run()."}])
        self.assertEqual(request["tools"][0]["name"], "web_search")
        self.assertEqual(request["tools"][0]["input_schema"], agent.WEB_SEARCH_TOOL["parameters"])
        # Adaptive thinking steered by effort: the form current Claude models accept.
        self.assertEqual(request["thinking"], {"type": "adaptive"})
        self.assertEqual(request["extra_body"], {"output_config": {"effort": "medium"}})
        self.assertEqual(request["max_tokens"], 4000 + claude_client.ADAPTIVE_HEADROOM)
        # Response in the Responses shape agent.py reads.
        calls = [item for item in response.output if item.type == "function_call"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].call_id, "toolu_01")
        self.assertEqual(calls[0].name, "web_search")
        self.assertEqual(json.loads(calls[0].arguments), {"query": "servicenow cdm publish snapshot version api"})
        self.assertEqual(response.output_text, "")
        self.assertEqual((response.usage.input_tokens, response.usage.output_tokens), (1000, 40))

    def test_second_turn_rebuilds_tool_use_with_thinking_and_tool_result(self):
        fake = FakeAnthropic([SEARCH_TURN, ANSWER_TURN])
        client = claude_client.ClaudeResponsesClient(api_key="k", anthropic_client=fake)
        conversation = [{"role": "system", "content": "S"}, {"role": "user", "content": "Q"}]
        first = client.responses.create(model="m", reasoning={"effort": "medium"}, tools=[agent.WEB_SEARCH_TOOL], input=conversation)
        agent._append_model_output(conversation, first)
        conversation.append(agent._tool_output(first.output[1], {"backend": "x", "query": "q", "hits": []}))
        second = client.responses.create(model="m", reasoning={"effort": "medium"}, tools=[agent.WEB_SEARCH_TOOL], input=conversation)
        request = fake.messages.requests[1]
        self.assertEqual(request["messages"][0], {"role": "user", "content": "Q"})
        assistant = request["messages"][1]
        self.assertEqual(assistant["role"], "assistant")
        self.assertEqual(assistant["content"][0], {"type": "thinking", "thinking": "I should search the docs.", "signature": "sig-1"})
        self.assertEqual(assistant["content"][1], {"type": "tool_use", "id": "toolu_01", "name": "web_search",
                                                   "input": {"query": "servicenow cdm publish snapshot version api"}})
        result = request["messages"][2]
        self.assertEqual(result["role"], "user")
        self.assertEqual(result["content"][0]["type"], "tool_result")
        self.assertEqual(result["content"][0]["tool_use_id"], "toolu_01")
        self.assertEqual(json.loads(result["content"][0]["content"]), {"backend": "x", "query": "q", "hits": []})
        self.assertEqual(len(request["messages"]), 3)
        self.assertTrue(second.output_text.startswith("```python"))
        self.assertFalse([item for item in second.output if item.type == "function_call"])

    def test_thinking_store_resets_on_a_new_conversation(self):
        fake = FakeAnthropic([SEARCH_TURN, SEARCH_TURN])
        client = claude_client.ClaudeResponsesClient(api_key="k", anthropic_client=fake)
        client.responses.create(model="m", reasoning={"effort": "low"}, input=[{"role": "user", "content": "A"}])
        self.assertIn("toolu_01", client._thinking)
        client.responses.create(model="m", reasoning={"effort": "low"}, input=[{"role": "user", "content": "B"}])
        self.assertEqual(fake.messages.requests[1]["extra_body"], {"output_config": {"effort": "low"}})

    def test_thinking_modes(self):
        # adaptive (default): thinking adaptive + output_config.effort, headroom on max_tokens
        with patch.dict(os.environ, {"CLAUDE_THINKING": "", "CLAUDE_THINKING_BUDGET": ""}):
            req = claude_client.thinking_request({"effort": "high"}, 4000)
            self.assertEqual(req, {"max_tokens": 4000 + claude_client.ADAPTIVE_HEADROOM, "thinking": {"type": "adaptive"},
                                   "extra_body": {"output_config": {"effort": "high"}}})
            # no reasoning asked → no thinking at all
            self.assertEqual(claude_client.thinking_request(None, 16), {"max_tokens": 16})
        # budget: the older fixed-budget form
        with patch.dict(os.environ, {"CLAUDE_THINKING": "budget", "CLAUDE_THINKING_BUDGET": ""}):
            self.assertEqual(claude_client.thinking_request({"effort": "medium"}, 4000),
                             {"max_tokens": 8096, "thinking": {"type": "enabled", "budget_tokens": 4096}})
        with patch.dict(os.environ, {"CLAUDE_THINKING": "budget", "CLAUDE_THINKING_BUDGET": "2000"}):
            self.assertEqual(claude_client.thinking_request({"effort": "medium"}, 4000)["thinking"]["budget_tokens"], 2000)
        # off, both spellings
        for envs in ({"CLAUDE_THINKING": "off", "CLAUDE_THINKING_BUDGET": ""}, {"CLAUDE_THINKING": "", "CLAUDE_THINKING_BUDGET": "0"}):
            with patch.dict(os.environ, envs):
                fake = FakeAnthropic([ANSWER_TURN])
                client = claude_client.ClaudeResponsesClient(api_key="k", anthropic_client=fake)
                client.responses.create(model="m", reasoning={"effort": "medium"}, input=[{"role": "user", "content": "A"}], max_output_tokens=500)
                self.assertNotIn("thinking", fake.messages.requests[0])
                self.assertNotIn("extra_body", fake.messages.requests[0])
                self.assertEqual(fake.messages.requests[0]["max_tokens"], 500)

    def test_multiple_tool_calls_in_one_turn_group_into_one_message_each(self):
        turn = [
            _block(type="tool_use", id="t1", name="web_search", input={"query": "a"}),
            _block(type="tool_use", id="t2", name="web_search", input={"query": "b"}),
        ]
        fake = FakeAnthropic([turn, ANSWER_TURN])
        client = claude_client.ClaudeResponsesClient(api_key="k", anthropic_client=fake)
        conversation = [{"role": "user", "content": "Q"}]
        first = client.responses.create(model="m", input=conversation)
        agent._append_model_output(conversation, first)
        for call in first.output:
            conversation.append(agent._tool_output(call, {"hits": []}))
        client.responses.create(model="m", input=conversation)
        messages = fake.messages.requests[1]["messages"]
        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "user"])
        self.assertEqual([b["id"] for b in messages[1]["content"]], ["t1", "t2"])
        self.assertEqual([b["tool_use_id"] for b in messages[2]["content"]], ["t1", "t2"])


class ProviderSelection(unittest.TestCase):
    def test_env_routes_to_claude_only_when_asked(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "k", "OPENAI_API_KEY": "o"}):
            self.assertEqual(env.llm_transport(), "anthropic")
            self.assertEqual(env.default_model(), claude_client.DEFAULT_CLAUDE_MODEL)
            self.assertIsInstance(env.make_llm_client(), claude_client.ClaudeResponsesClient)
        with patch.dict(os.environ, {"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "k", "CLAUDE_MODEL": "claude-opus-5-5"}):
            self.assertEqual(env.default_model(), "claude-opus-5-5")
        with patch.dict(os.environ, {"LLM_PROVIDER": "", "ANTHROPIC_API_KEY": "k", "OPENAI_API_KEY": "o"}):
            self.assertEqual(env.llm_transport(), "openai")
            self.assertEqual(env.default_model(), agent.DEFAULT_MODEL)
        with patch.dict(os.environ, {"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": ""}):
            self.assertEqual(env.llm_transport(), "missing")
            with self.assertRaises(SystemExit):
                env.make_llm_client()


class EndToEndThroughClaude(unittest.TestCase):
    def test_agent_loop_runs_unmodified_and_the_judge_passes(self):
        fake = FakeAnthropic([SEARCH_TURN, ANSWER_TURN])
        client = claude_client.ClaudeResponsesClient(api_key="k", anthropic_client=fake)
        envs = {"JEVLIN_API_KEY": "test-key", "JEVLIN_PROVIDERS": "exa,brave,serpapi,tinyfish,you"}
        with patch.dict(os.environ, envs), patch.object(search, "vendor_call", return_value=(MERGED, {"http_status": 200, "http_ms": 800})):
            run = agent.run_agent(
                TASK.prompt, client=client, backend_name="jevlin_fanout_merged", model="claude-test",
                max_turns=6, max_searches=5, max_fetches=0, system=harness_eval.PATCH_SYSTEM_SEARCH_ONLY,
                allow_search=True, allow_fetch=False, leak_needles=TASK.leak_needles(),
            )
        source = judge.extract_code(run.answer, TASK.outfile)
        score = judge.score_needles(source, TASK, observed_urls=judge.hop_urls(run), retrieved_text=judge.hop_retrieved_text(run))
        self.assertTrue(score.passed, score.output)
        self.assertEqual(len(run.searches), 1)
        self.assertEqual(run.searches[0].query, "servicenow cdm publish snapshot version api")
        self.assertEqual(run.llm_prompt_tokens, 2000)
        self.assertEqual(len(run.llm_turns), 2)
        self.assertEqual(run.llm_turns[0].tool_names, ["web_search"])
        # The second request carried the first turn's thinking and tool_use, then the tool_result the agent built.
        second = fake.messages.requests[1]
        self.assertEqual(second["messages"][1]["content"][0]["type"], "thinking")
        self.assertEqual(second["messages"][1]["content"][1]["type"], "tool_use")
        self.assertEqual(second["messages"][2]["content"][0]["type"], "tool_result")
        self.assertIn("hits", json.loads(second["messages"][2]["content"][0]["content"]))


if __name__ == "__main__":
    unittest.main()
