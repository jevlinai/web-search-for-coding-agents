"""The headless-Claude-Code driver, offline, against a fake `claude` binary.

The fake reads the allowed tool pattern from its arguments, runs the generated web_search
command once (which calls cc_search.py, which serves the merged fixture for the jevlin_fixture
arm), and prints a Claude Code-shaped JSON result with the gold file. A second fake answers with
a rate-limit error first and succeeds on retry; a third never calls the tool and returns prose.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from coding_search.tasks import CodingTask
from coding_search.test_jevlin_e2e import GOLD

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIXTURE = HERE / "testdata" / "jevlin" / "merged_response.json"
sys.path.insert(0, str(ROOT / "scripts" / "jevlin_bench"))
import cc_run  # noqa: E402

FAKE_CLAUDE = r'''#!/bin/bash
# Fake claude: find the wrapper in --allowedTools "Bash(<path> *)", search once, answer with the gold file.
# Paths are baked in: the driver starts claude with an empty environment, as it must.
MODE_FILE="@@MODE@@"
GOLD_FILE="@@GOLD@@"
wrapper=""
prev=""
for a in "$@"; do
  if [ "$prev" = "--allowedTools" ]; then wrapper="${a#Bash(}"; wrapper="${wrapper% \*)}"; fi
  prev="$a"
done
if [ "$1" = "--version" ]; then echo "9.9.9 (Claude Code fake)"; exit 0; fi
# the session must not see an API key or JEVLIN_* variables
if [ -n "${ANTHROPIC_API_KEY:-}" ] || [ -n "${JEVLIN_CONFIG:-}" ] || [ -n "${CLAUDECODE:-}" ]; then echo '{"type":"result","is_error":true,"result":"leaked env"}'; exit 0; fi
mode=$(cat "$MODE_FILE" 2>/dev/null || echo ok)
if [ "$mode" = "ratelimit-once" ]; then
  echo "ok" > "$MODE_FILE"
  echo '{"type":"result","subtype":"error","is_error":true,"result":"You have hit your limit · resets at 3pm","num_turns":0,"usage":{"input_tokens":0,"output_tokens":0}}'
  exit 1
fi
if [ "$mode" = "no-search" ]; then
  printf '{"type":"result","subtype":"success","is_error":false,"result":"I would need to search but cannot.","num_turns":1,"duration_ms":10,"duration_api_ms":5,"session_id":"s0","total_cost_usd":0.001,"usage":{"input_tokens":10,"output_tokens":5}}\n'
  exit 0
fi
"$wrapper" "servicenow cdm publish snapshot version api" > "$(dirname "$wrapper")/shown.json"
"$wrapper" "second query" > /dev/null
python3 - "$GOLD_FILE" <<'PY'
import json, sys
gold = open(sys.argv[1]).read()
print(json.dumps({"type": "result", "subtype": "success", "is_error": False,
                  "result": "```python\n" + gold + "```", "num_turns": 4, "duration_ms": 1234, "duration_api_ms": 900,
                  "session_id": "sess_fake", "total_cost_usd": 0.0123,
                  "usage": {"input_tokens": 800, "cache_creation_input_tokens": 100, "cache_read_input_tokens": 100, "output_tokens": 60},
                  "modelUsage": {"claude-test": {"inputTokens": 1000, "outputTokens": 60}},
                  "permission_denials": []}))
PY
'''


class Driver(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.mode_file = root / "mode"
        self.mode_file.write_text("ok")
        self.gold_file = root / "gold.py"
        self.gold_file.write_text(GOLD)
        self.fake = root / "claude"
        self.fake.write_text(FAKE_CLAUDE.replace("@@MODE@@", str(self.mode_file)).replace("@@GOLD@@", str(self.gold_file)))
        self.fake.chmod(0o755)
        # a one-ticket dataset tree
        tasks = root / "ds" / "web-search" / "coding" / "tasks"
        task = tasks / "fixture_sn_publish"
        (task / "gold").mkdir(parents=True)
        (task / "starter").mkdir(parents=True)
        (task / "gold" / "file.py").write_text(GOLD)
        (task / "starter" / "file.py").write_text("import httpx\n")
        (task / "prompt.txt").write_text("Patch run() so the snapshot version is published.\n\nCurrent `file.py`:\n\n```python\nimport httpx\n```\n")
        (tasks / "catalog.json").write_text(json.dumps({"defaults": {"outfile": "file.py", "cite_urls": True}, "items": [
            {"id": "fixture_sn_publish", "repo": "servicenow", "required": ["sn_cdm/versions/publish"],
             "query_forbidden": ["sn_cdm/versions/publish"], "docs": ["https://www.servicenow.com/docs/r/api-reference/rest-apis/versions-api.html"]}]}))
        self.root = root
        # The parent process carries secrets and a parent-session marker; none may reach the fake claude.
        self.env = {"DATASETS_ROOT": str(root / "ds"), "ANTHROPIC_API_KEY": "must-not-leak",
                    "JEVLIN_CONFIG": "/home/op/.jevlin/jevlin.toml", "CLAUDECODE": "1"}

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, *extra):
        out = self.root / "out" / "pass-1.json"
        argv = ["--arms", "jevlin_fixture", "--ids", "fixture_sn_publish", "--seed", "1", "--out", str(out),
                "--raw-dir", str(self.root / "out" / "raw"), "--jobs-dir", str(self.root / "out" / "jobs"),
                "--claude-bin", str(self.fake), "--fixture", str(FIXTURE), "--rate-limit-wait", "0", "--model", "claude-test", *extra]
        with unittest_env(self.env):
            code = cc_run.main(argv)
        return code, json.loads(out.read_text())

    def test_one_job_end_to_end(self):
        code, summary = self._run()
        self.assertEqual(code, 0)
        self.assertEqual(summary["n"], 1)
        self.assertEqual(summary["passed"], 1)
        self.assertEqual(summary["agent"]["kind"], "claude-code")
        row = summary["results"][0]
        self.assertTrue(row["passed"], row["score"])
        self.assertIsNone(row["error"])
        self.assertEqual(row["search_backend"], "jevlin_fixture")
        self.assertEqual(row["search_count"], 2)
        self.assertEqual(row["queries"][0], "servicenow cdm publish snapshot version api")
        self.assertEqual(row["search_hops"][0]["hit_urls"][0], "https://servicenow.com/docs/r/api-reference/rest-apis/versions-api.html")
        self.assertNotIn("response", row["search_hops"][0])
        self.assertEqual(row["metrics"]["llm_prompt_tokens"], 1000)
        self.assertEqual(row["metrics"]["llm_completion_tokens"], 60)
        self.assertEqual(row["agent"]["num_turns"], 4)
        self.assertEqual(row["agent"]["total_cost_usd"], 0.0123)
        self.assertEqual(row["agent"]["version"], "9.9.9 (Claude Code fake)")
        # the prompt names the command by absolute path, once per run
        self.assertIn(str(self.root / "out" / "jobs" / "bin" / "web_search"), (Path(row["agent"]["job_dir"]) / "system.txt").read_text())
        # what the model was shown: the harness tool-output shape
        shown = json.loads((self.root / "out" / "jobs" / "bin" / "shown.json").read_text())
        self.assertEqual(set(shown), {"backend", "query", "hits"})
        self.assertEqual(set(shown["hits"][0]), {"url", "title", "snippet"})
        # raw dumps in the harness layout, with the gateway body
        hop = Path(row["raw_dir"]) / "hops" / "search-01.response.json"
        self.assertTrue(hop.is_file())
        self.assertEqual(json.loads(hop.read_text())["body"]["request_id"], "req_fixture_0001")

    def test_relative_paths_are_made_absolute_for_prompt_and_allow_rule(self):
        cwd = os.getcwd()
        os.chdir(self.root)
        try:
            out = Path("rel-out") / "pass-1.json"
            argv = ["--arms", "jevlin_fixture", "--ids", "fixture_sn_publish", "--out", str(out), "--jobs-dir", "rel-out/jobs",
                    "--raw-dir", "rel-out/raw", "--claude-bin", str(self.fake), "--fixture", str(FIXTURE), "--model", "claude-test"]
            with unittest_env(self.env):
                code = cc_run.main(argv)
            summary = json.loads((self.root / out).read_text())
        finally:
            os.chdir(cwd)
        self.assertEqual(code, 0)
        row = summary["results"][0]
        self.assertTrue(row["passed"], row["score"])
        system = (Path(row["agent"]["job_dir"]) / "system.txt").read_text()
        self.assertIn(str(self.root / "rel-out" / "jobs" / "bin" / "web_search"), system)
        self.assertTrue(Path(row["raw_dir"]).is_absolute())
        self.assertIn(str(self.root / "rel-out" / "jobs" / "bin" / "web_search"), summary["agent"]["system_prompt"])

    def test_rate_limit_is_waited_out_and_retried(self):
        self.mode_file.write_text("ratelimit-once")
        code, summary = self._run("--rate-limit-retries", "2")
        self.assertEqual(code, 0)
        row = summary["results"][0]
        self.assertTrue(row["passed"], row["score"])
        self.assertEqual(row["agent"]["attempts"], 2)

    def test_rate_limit_exhausted_is_a_run_error_and_aborts(self):
        self.mode_file.write_text("ratelimit-once")
        code, summary = self._run("--rate-limit-retries", "0", "--abort-after-run-errors", "1")
        self.assertEqual(code, 1)
        self.assertTrue(summary["aborted"])
        row = summary["results"][0]
        self.assertFalse(row["passed"])
        self.assertIn("claude error", row["error"])

    def test_a_run_of_failed_searches_aborts_the_pass(self):
        # A missing fixture makes every search of the jevlin_fixture arm raise: a 429 storm stand-in.
        out = self.root / "out" / "pass-1.json"
        argv = ["--arms", "jevlin_fixture", "--ids", "fixture_sn_publish", "--out", str(out), "--jobs-dir", str(self.root / "out" / "jobs"),
                "--claude-bin", str(self.fake), "--fixture", str(self.root / "missing-fixture.json"), "--model", "claude-test",
                "--abort-after-hop-errors", "2"]
        with unittest_env(self.env):
            code = cc_run.main(argv)
        summary = json.loads(out.read_text())
        self.assertEqual(code, 1)
        self.assertTrue(summary["aborted"])
        row = summary["results"][0]
        self.assertIsNone(row["error"])  # the job itself ran; its searches failed
        self.assertEqual(row["search_count"], 2)
        self.assertTrue(all(h["error"] for h in row["search_hops"]))
        self.assertFalse(row["passed"])

    def test_no_search_fails_the_judge_with_a_note(self):
        self.mode_file.write_text("no-search")
        code, summary = self._run()
        self.assertEqual(code, 0)
        row = summary["results"][0]
        self.assertFalse(row["passed"])
        self.assertIsNone(row["error"])
        self.assertIn("no search was made", row["notes"])
        self.assertEqual(row["search_count"], 0)

    def test_budget_is_enforced_by_the_wrapper(self):
        hop_dir = self.root / "hops"
        hop_dir.mkdir()
        wrapper = cc_run.write_wrapper(self.root / "tools", arm="jevlin_fixture", hop_dir=hop_dir, max_searches=2,
                                       max_results=8, python=sys.executable, fixture=str(FIXTURE))
        outputs = [subprocess.run([str(wrapper), f"query {i}"], capture_output=True, text=True, check=False).stdout for i in range(3)]
        self.assertEqual(len(list(hop_dir.glob("search-*.json"))), 2)
        self.assertIn("hits", json.loads(outputs[0]))
        self.assertEqual(json.loads(outputs[2]), {"error": "search budget exhausted; answer now"})
        self.assertTrue((hop_dir / "budget_exhausted").exists())
        usage = subprocess.run([str(wrapper)], capture_output=True, text=True, check=False)
        self.assertEqual(usage.returncode, 64)

    def test_clean_env_has_no_secrets_or_parent_markers(self):
        with unittest_env({"ANTHROPIC_API_KEY": "x", "JEVLIN_API_KEY": "y", "CLAUDECODE": "1", "JEVLIN_CONFIG": "z"}):
            env = cc_run.clean_env(str(self.fake))
        for key in ("ANTHROPIC_API_KEY", "JEVLIN_API_KEY", "JEVLIN_CONFIG", "CLAUDECODE"):
            self.assertNotIn(key, env)
        self.assertIn("HOME", env)
        self.assertIn(str(self.fake.parent), env["PATH"])
        tools = self.root / "tools"
        self.assertTrue(cc_run.clean_env(str(self.fake), tools_dir=tools)["PATH"].startswith(str(tools) + ":"))

    def test_both_spellings_of_the_command_are_allowed(self):
        argv = cc_run.claude_argv(str(self.fake), prompt="p", system="s", wrapper=self.root / "tools" / "web_search", model="m")
        i = argv.index("--allowedTools")
        self.assertEqual(argv[i + 1], f"Bash({self.root / 'tools' / 'web_search'} *)")
        self.assertEqual(argv[i + 2], "Bash(web_search *)")
        self.assertIn("--safe-mode", argv)
        self.assertIn("WebSearch,WebFetch,Read,Write,Edit,Glob,Grep,Task,TodoWrite,NotebookEdit", argv)


class unittest_env:
    """Temporarily set environment variables."""

    def __init__(self, values: dict[str, str]):
        self.values = values
        self.saved: dict[str, str | None] = {}

    def __enter__(self):
        for key, value in self.values.items():
            self.saved[key] = os.environ.get(key)
            os.environ[key] = value
        return self

    def __exit__(self, *exc):
        for key, old in self.saved.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old
        return False


if __name__ == "__main__":
    unittest.main()
