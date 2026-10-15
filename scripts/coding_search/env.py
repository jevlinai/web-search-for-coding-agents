"""Load secrets without printing them."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATASETS_DEFAULT = ROOT / "datasets"

AZURE_KEY_ENV = "AZURE_OPENAI_NEXTGEN_DEPLOYMENT_KEY"
AZURE_URL_ENV = "AZURE_OPENAI_NEXTGEN_DEPLOYMENT_URL"
# Benchmark fork: LLM_PROVIDER=anthropic routes the unchanged agent loop to a Claude
# model through coding_search.claude_client. Explicit, so a stray key never flips it.
ANTHROPIC_KEY_ENV = "ANTHROPIC_API_KEY"
PROVIDER_ENV = "LLM_PROVIDER"


def anthropic_selected() -> bool:
    return (os.environ.get(PROVIDER_ENV) or "").strip().lower() == "anthropic"


def default_model() -> str:
    """The model a run uses when --model is not given: the board's, or the Claude default."""
    if anthropic_selected():
        from .claude_client import DEFAULT_CLAUDE_MODEL

        return (os.environ.get("CLAUDE_MODEL") or DEFAULT_CLAUDE_MODEL).strip()
    from .agent import DEFAULT_MODEL

    return DEFAULT_MODEL


def load_environment() -> None:
    load_dotenv(ROOT / ".env.local")
    load_dotenv(ROOT / ".env")


def datasets_root() -> Path:
    raw = os.environ.get("DATASETS_ROOT")
    if raw:
        return Path(raw).expanduser().resolve()
    return DATASETS_DEFAULT


def llm_transport() -> str:
    if anthropic_selected():
        return "anthropic" if os.environ.get(ANTHROPIC_KEY_ENV) else "missing"
    if os.environ.get(AZURE_URL_ENV) and os.environ.get(AZURE_KEY_ENV):
        return "azure"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "missing"


def make_llm_client(*, timeout: float = 180, max_retries: int = 1) -> Any:
    """Azure Foundry OpenAI-v1 deployment when NEXTGEN env is set, else native OpenAI.

    With LLM_PROVIDER=anthropic: a Claude client presenting the same `responses.create`
    surface (benchmark fork).
    """
    if anthropic_selected():
        key = os.environ.get(ANTHROPIC_KEY_ENV) or ""
        if not key:
            raise SystemExit(f"{PROVIDER_ENV}=anthropic needs {ANTHROPIC_KEY_ENV}")
        from .claude_client import ClaudeResponsesClient

        return ClaudeResponsesClient(api_key=key, timeout=timeout, max_retries=max_retries)

    from openai import OpenAI

    azure_url = (os.environ.get(AZURE_URL_ENV) or "").rstrip("/")
    azure_key = os.environ.get(AZURE_KEY_ENV) or ""
    if azure_url and azure_key:
        return OpenAI(
            base_url=azure_url,
            api_key=azure_key,
            timeout=timeout,
            max_retries=max_retries,
        )
    key = os.environ.get("OPENAI_API_KEY") or ""
    if not key:
        raise SystemExit(
            f"set {AZURE_URL_ENV} + {AZURE_KEY_ENV} (preferred) or OPENAI_API_KEY"
        )
    return OpenAI(api_key=key, timeout=timeout, max_retries=max_retries)
