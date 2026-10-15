"""Claude behind the harness's OpenAI-Responses-shaped client. Benchmark fork only.

`agent.py` speaks the OpenAI Responses API: it calls
`client.responses.create(model, reasoning, tools, input, max_output_tokens, store)`
and reads `response.output` (function_call items), `response.output_text` and
`response.usage`. This module presents exactly that surface over Anthropic's
Messages API, so the agent loop, its budgets, the prompts and the judge run
unmodified with a Claude model. Nothing in `agent.py` changes.

Translation:
  system message              → `system`
  user / assistant text       → messages
  function_call items         → one assistant message of `tool_use` blocks per turn
  function_call_output items  → one user message of `tool_result` blocks per turn
  OpenAI function tools       → Anthropic tools (`input_schema` = `parameters`)
  reasoning.effort            → adaptive thinking steered by `output_config.effort`
                                (CLAUDE_THINKING=adaptive, the default; `budget` for the
                                older fixed-budget form; `off` to disable)
  max_output_tokens           → `max_tokens`, raised by the thinking headroom or budget
  tool_use blocks             → function_call items (call_id = the block id)
  text blocks                 → `output_text`

Anthropic requires the thinking blocks of an assistant turn that used tools to
be sent back with that turn. `agent.py` drops reasoning items from the
conversation it keeps, so this client remembers each turn's thinking blocks by
tool-use id and re-attaches them when it rebuilds the assistant message. The
store is cleared whenever a conversation starts over (no function calls in it).
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Any

DEFAULT_CLAUDE_MODEL = "claude-sonnet-5-5"
EFFORT_BUDGET = {"low": 1024, "medium": 4096, "high": 8192}
MIN_BUDGET = 1024
EFFORTS = ("low", "medium", "high")
# Thinking modes. `adaptive` is what current Claude models accept: the model decides how
# much to think and `output_config.effort` steers it, the direct analogue of the harness's
# `reasoning.effort`. `budget` is the older fixed-budget form for models that still take it.
# CLAUDE_THINKING selects; CLAUDE_THINKING_BUDGET=0 is a legacy spelling of `off`.
THINKING_MODES = ("adaptive", "budget", "off")
# With adaptive thinking the thinking tokens share max_tokens with the visible answer, so
# the harness's visible cap is raised by this much to keep the answer from being starved.
ADAPTIVE_HEADROOM = 8192


def thinking_mode() -> str:
    if (os.environ.get("CLAUDE_THINKING_BUDGET") or "").strip() == "0":
        return "off"
    raw = (os.environ.get("CLAUDE_THINKING") or "adaptive").strip().lower()
    return raw if raw in THINKING_MODES else "adaptive"


def thinking_budget(reasoning: dict[str, Any] | None) -> int:
    raw = (os.environ.get("CLAUDE_THINKING_BUDGET") or "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = 0
        return 0 if value <= 0 else max(MIN_BUDGET, value)
    effort = str((reasoning or {}).get("effort") or "").lower()
    return EFFORT_BUDGET.get(effort, 0) if effort else 0


def thinking_request(reasoning: dict[str, Any] | None, max_output_tokens: int) -> dict[str, Any]:
    """The request fields that carry the harness's reasoning setting to Claude."""
    effort = str((reasoning or {}).get("effort") or "").lower()
    mode = thinking_mode()
    if not effort or mode == "off":
        return {"max_tokens": int(max_output_tokens)}
    if mode == "budget":
        budget = thinking_budget(reasoning)
        if not budget:
            return {"max_tokens": int(max_output_tokens)}
        return {"max_tokens": int(max_output_tokens) + budget, "thinking": {"type": "enabled", "budget_tokens": budget}}
    return {
        "max_tokens": int(max_output_tokens) + ADAPTIVE_HEADROOM,
        "thinking": {"type": "adaptive"},
        "extra_body": {"output_config": {"effort": effort if effort in EFFORTS else "medium"}},
    }


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            text = _get(part, "text")
            if text:
                parts.append(str(text))
        return "\n".join(parts)
    return str(content or "")


def convert_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tool in tools or []:
        if _get(tool, "type") != "function":
            continue
        out.append({
            "name": str(_get(tool, "name")),
            "description": str(_get(tool, "description") or ""),
            "input_schema": dict(_get(tool, "parameters") or {"type": "object", "properties": {}}),
        })
    return out


def _arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def convert_input(
    items: list[Any],
    thinking_for: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[str | None, list[dict[str, Any]]]:
    """OpenAI Responses input items → (system, Anthropic messages)."""
    system_parts: list[str] = []
    messages: list[dict[str, Any]] = []
    tool_uses: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []

    def flush_uses() -> None:
        if not tool_uses:
            return
        blocks: list[dict[str, Any]] = []
        for block in tool_uses:
            remembered = (thinking_for or {}).get(block["id"])
            if remembered:
                blocks = list(remembered)
                break
        blocks.extend(tool_uses)
        messages.append({"role": "assistant", "content": blocks})
        tool_uses.clear()

    def flush_results() -> None:
        if tool_results:
            messages.append({"role": "user", "content": list(tool_results)})
            tool_results.clear()

    for item in items:
        kind = _get(item, "type") or ""
        role = _get(item, "role") or ""
        if kind == "function_call":
            flush_results()
            tool_uses.append({
                "type": "tool_use",
                "id": str(_get(item, "call_id") or _get(item, "id") or ""),
                "name": str(_get(item, "name") or ""),
                "input": _arguments(_get(item, "arguments")),
            })
            continue
        if kind == "function_call_output":
            flush_uses()
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": str(_get(item, "call_id") or ""),
                "content": str(_get(item, "output") or ""),
            })
            continue
        flush_uses()
        flush_results()
        text = _text_of(_get(item, "content"))
        if role == "system" or role == "developer":
            system_parts.append(text)
        elif role in ("user", "assistant"):
            messages.append({"role": role, "content": text})
        elif kind == "message":
            messages.append({"role": str(role or "assistant"), "content": text})
        # reasoning items and anything else: nothing to send
    flush_uses()
    flush_results()
    return ("\n\n".join(part for part in system_parts if part) or None), messages


def _thinking_block(block: Any) -> dict[str, Any] | None:
    kind = _get(block, "type")
    if kind == "thinking":
        return {"type": "thinking", "thinking": str(_get(block, "thinking") or ""), "signature": str(_get(block, "signature") or "")}
    if kind == "redacted_thinking":
        return {"type": "redacted_thinking", "data": str(_get(block, "data") or "")}
    return None


class ClaudeResponsesClient:
    """Duck-types the slice of `openai.OpenAI` that `agent.py` uses."""

    def __init__(self, *, api_key: str, timeout: float = 180, max_retries: int = 1, anthropic_client: Any = None):
        if anthropic_client is None:
            import anthropic

            anthropic_client = anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=max_retries)
        self._client = anthropic_client
        self._thinking: dict[str, list[dict[str, Any]]] = {}
        self.responses = SimpleNamespace(create=self.create)
        self.last_request: dict[str, Any] | None = None

    def create(
        self,
        *,
        model: str,
        input: list[Any],
        tools: list[dict[str, Any]] | None = None,
        reasoning: dict[str, Any] | None = None,
        max_output_tokens: int = 4000,
        **_ignored: Any,
    ) -> Any:
        if not any(_get(item, "type") == "function_call" for item in input):
            self._thinking.clear()
        system, messages = convert_input(input, self._thinking)
        request: dict[str, Any] = {
            "model": model,
            "messages": messages,
            **thinking_request(reasoning, max_output_tokens),
        }
        if system:
            request["system"] = system
        converted = convert_tools(tools)
        if converted:
            request["tools"] = converted
        self.last_request = request
        response = self._client.messages.create(**request)
        return self._to_responses(response)

    def _to_responses(self, response: Any) -> Any:
        output: list[Any] = []
        texts: list[str] = []
        thinking: list[dict[str, Any]] = []
        call_ids: list[str] = []
        for block in _get(response, "content") or []:
            kind = _get(block, "type")
            if kind in ("thinking", "redacted_thinking"):
                kept = _thinking_block(block)
                if kept:
                    thinking.append(kept)
                output.append(SimpleNamespace(type="reasoning", id=None, call_id=None, name=None, arguments=None, status="completed"))
            elif kind == "text":
                texts.append(str(_get(block, "text") or ""))
                output.append(SimpleNamespace(type="message", id=None, call_id=None, name=None, arguments=None, status="completed"))
            elif kind == "tool_use":
                call_id = str(_get(block, "id") or "")
                call_ids.append(call_id)
                output.append(SimpleNamespace(
                    type="function_call",
                    id=f"fc_{call_id}",
                    call_id=call_id,
                    name=str(_get(block, "name") or ""),
                    arguments=json.dumps(_get(block, "input") or {}, ensure_ascii=False),
                    status="completed",
                ))
        for call_id in call_ids:
            self._thinking[call_id] = thinking
        usage = _get(response, "usage")
        input_tokens = int(_get(usage, "input_tokens", 0) or 0)
        input_tokens += int(_get(usage, "cache_read_input_tokens", 0) or 0)
        input_tokens += int(_get(usage, "cache_creation_input_tokens", 0) or 0)
        output_tokens = int(_get(usage, "output_tokens", 0) or 0)
        return SimpleNamespace(
            output=output,
            output_text="\n".join(text for text in texts if text).strip(),
            usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
            stop_reason=_get(response, "stop_reason"),
            model=_get(response, "model"),
        )
