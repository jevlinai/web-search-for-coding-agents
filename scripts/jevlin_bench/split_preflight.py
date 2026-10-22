#!/usr/bin/env python3
"""Pre-flight for the split arm: does the router know `excerpt_mode`?

The router decodes `/v1/search` strictly, so a router that predates the split view answers a request
carrying `excerpt_mode` with `400 malformed_json` (an unknown field), and one that has it validates
the value and refuses a bad one with `400 invalid_request`. Without this check a run pointed at an
older host would send `jevlin_balanced_split` a request it refuses on every search, or, on a router
that ignores unknown fields, would run the arm without its split view and read the result as one.

The probe is one request the router refuses before any provider is called: `excerpt_mode:"split"` on
tier `fast`, a combination the split view does not allow (valid with view merged and tier balanced or
max). Which refusal comes back is the answer:

  400 invalid_request that names excerpt_mode   the router knows the field and the value: accepted
  400 malformed_json                            the router has no such field: refused
  200, or anything else                         no refusal could be read: refused. A 200 means the
                                                router ignored the field or does not enforce its
                                                tier rule, and it billed one fast-tier search.

On a router that has the field the probe is free. It is deliberately not a successful split search:
that would bill a balanced query and would not tell a router that has the field from one that ignores
it any better than a refusal does.

Importable (`probe`, `classify`, `require_split_router`) and runnable: `python split_preflight.py`
probes the router named by JEVLIN_API_URL with JEVLIN_API_KEY (read from the environment or the
fork's .env through the harness's loader; values are never printed).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import requests  # noqa: E402
from coding_search import jevlin  # noqa: E402

PROBE_QUERY = "what is the boiling point of water at sea level"
FIELD = "excerpt_mode"
# The router's refusal for a field its strict decoder does not know, and for a known field with an
# unacceptable value or combination.
UNKNOWN_FIELD_CODE = "malformed_json"
VALIDATION_CODE = "invalid_request"


def probe_body() -> dict[str, Any]:
    # Valid in every respect except the combination under test: split is not allowed on tier fast.
    return {"query": PROBE_QUERY, "tier": "fast", "view": "merged", "max_results": 1, FIELD: jevlin.EXCERPT_MODE_SPLIT}


def refusal_parts(payload: Any) -> tuple[str, str]:
    """(code, text) of a router error body, lower-cased. Flat `{"error", "code"}`; a nested error object is tolerated."""
    if not isinstance(payload, dict):
        return "", str(payload or "").lower()
    error = payload.get("error")
    nested = error if isinstance(error, dict) else {}
    code = str(payload.get("code") or nested.get("code") or "")
    message = nested.get("message") if nested else error
    text = " ".join(str(part) for part in (message, payload.get("message")) if part)
    return code.lower(), text.lower()


def classify(status_code: int | None, payload: Any) -> tuple[bool, str]:
    """(accepted, detail) for the probe's answer. Accepted only on a refusal that proves the router knows the field."""
    code, text = refusal_parts(payload)
    if status_code == 400 and code == UNKNOWN_FIELD_CODE:
        return False, f"400 {UNKNOWN_FIELD_CODE}: the router does not know {FIELD} (an unknown field)"
    if status_code == 400 and code == VALIDATION_CODE and FIELD in text:
        return True, f"accepted: 400 {VALIDATION_CODE} naming {FIELD}"
    if status_code == 200:
        return False, f"200: the router served {FIELD}=split on tier fast, so it ignored the field or does not enforce its tier rule (one fast-tier search was billed)"
    return False, f"unreadable refusal: HTTP {status_code} code={code or '-'}: {text[:200] or '-'}"


def probe() -> tuple[bool, str]:
    """(accepted, detail). One request, no retry: a refusal is the answer, not a fault to ride out."""
    key = (os.environ.get("JEVLIN_API_KEY") or "").strip()
    if not key:
        return False, "JEVLIN_API_KEY is required for the probe"
    try:
        # requests.post is looked up here, at call time, so a test that replaces it is never bypassed.
        response = requests.post(
            jevlin.api_url(),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=probe_body(),
            timeout=jevlin.timeout_s(),
        )
    except requests.RequestException as exc:
        return False, f"no answer: {type(exc).__name__}"
    try:
        payload: Any = response.json()
    except ValueError:
        payload = (response.text or "")[:500]
    return classify(response.status_code, payload)


def require_split_router(arms: list[str], probe_fn=None) -> str | None:
    """None when no arm needs the split view or the router refuses the probe the way a split router does;
    otherwise the refusal message.

    `probe` is looked up when this runs, not bound as a default when the module loads: a default of
    `probe` would keep the real function, so a test that patches `split_preflight.probe` would still
    make a live request.
    """
    if not any(jevlin.is_split_arm(arm) for arm in arms):
        return None
    accepted, detail = (probe_fn or probe)()
    if accepted:
        return None
    return (f"refusing to start: the router did not confirm {FIELD} ({detail}). "
            f"The split arm needs a router that validates {FIELD}; without it every search of the arm would be "
            f"refused, or would run without its split view and be read as a result. "
            f"The probe sends {FIELD}=split on tier fast, which a router with the field refuses before any provider is called.")


def main() -> int:
    from coding_search.env import load_environment

    load_environment()
    accepted, detail = probe()
    print(f"split pre-flight against {os.environ.get('JEVLIN_API_URL') or jevlin.DEFAULT_API_URL}: {detail}")
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
