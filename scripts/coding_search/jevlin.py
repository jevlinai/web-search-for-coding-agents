"""Jevlin backends for the coding-search harness. Benchmark-only code.

Jevlin is a multi-provider search gateway: one `POST /v1/search` fans out to a
tier-defined set of upstream providers and returns every provider's results
side by side; with `view:"merged"` it also returns one deduplicated list of
pages ordered by how many providers returned each page. These backends
translate that into the harness's url / title / snippet hits and keep, in the
hop metadata, what the harness's own fields cannot say: which upstream
providers ran, what each cost, and whether Jevlin's cache answered instead.

Nothing here changes Jevlin. Each backend is one request shape against the
public API, or one invocation of the public `jevlin` CLI.

Every backend shows the model at most `max_results` hits of url / title /
snippet (snippet cut at 1200 characters), exactly like every other vendor in
the harness. The same `max_results` is sent to Jevlin as its per-provider
result count, so a single-provider arm asks a provider for the same number of
results the direct runner would.

Arms, and the experiment each serves:

| backend                | request                                   | model sees                        |
|------------------------|-------------------------------------------|-----------------------------------|
| jevlin_fast            | tier=fast, tenant's default providers     | Jevlin's merged order, top N      |
| jevlin_balanced        | tier=balanced, default providers          | merged order, top N               |
| jevlin_max             | tier=max, default providers               | merged order, top N               |
| jevlin_single          | providers=[JEVLIN_SINGLE_PROVIDER]        | that provider's own order, top N  |
| jevlin_fanout_raw      | tier=balanced, providers=JEVLIN_PROVIDERS | round-robin over providers, exact-URL dedup, top N |
| jevlin_fanout_merged   | tier=balanced, providers=JEVLIN_PROVIDERS | merged order, top N               |
| jevlin_cli             | `jevlin search --stdin`, no tier, no view | the CLI's own client-side merge, top N |

`jevlin_fanout_*` refuse to run without `JEVLIN_PROVIDERS`: the ablation must
pin the set so a configuration change on the server cannot move it between
passes. The tier arms deliberately do not pin, because they measure what a
customer gets.

Environment:
  JEVLIN_API_KEY          bearer key for the HTTP arms (a portal key; agent
                          keys are restricted to the fast tier)
  JEVLIN_API_URL          base URL, default https://api.jevlin.ai
  JEVLIN_PROVIDERS        comma list, required by jevlin_fanout_*
  JEVLIN_SINGLE_PROVIDER  provider name for jevlin_single, default exa
  JEVLIN_BIN              path to the jevlin executable for jevlin_cli
  JEVLIN_TIMEOUT_S        HTTP / CLI timeout, default 45 (Jevlin's own
                          fan-out deadline is 30 s in the deployed config)
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from typing import Any

DEFAULT_API_URL = "https://api.jevlin.ai"
SEARCH_PATH = "/v1/search"
DEFAULT_TIMEOUT_S = 45.0
SNIPPET_CHARS = 1200
# Statuses the gateway assigns an arm it never called. Everything else was an
# upstream request, billed or not.
NOT_CALLED = frozenset({"skipped"})


def _harness():
    # Lazy: search.py imports this module to register the backends, and this
    # module needs search.py's HTTP wrapper and hit-stats helpers.
    from . import search

    return search


def api_url() -> str:
    base = (os.environ.get("JEVLIN_API_URL") or DEFAULT_API_URL).strip().rstrip("/")
    return base + SEARCH_PATH


def timeout_s() -> float:
    raw = (os.environ.get("JEVLIN_TIMEOUT_S") or "").strip()
    try:
        return float(raw) if raw else DEFAULT_TIMEOUT_S
    except ValueError:
        return DEFAULT_TIMEOUT_S


def pinned_providers() -> list[str]:
    raw = os.environ.get("JEVLIN_PROVIDERS") or ""
    out: list[str] = []
    for item in raw.split(","):
        name = item.strip()
        if name and name not in out:
            out.append(name)
    return out


# ── presentation: Jevlin response → harness hits ───────────────────────────


def _hit(url: str, title: Any, snippet: Any) -> dict[str, str]:
    return {"url": url, "title": str(title or ""), "snippet": str(snippet or "")[:SNIPPET_CHARS]}


def present_merged(payload: Any, *, max_results: int) -> list[dict[str, str]]:
    """Jevlin's `merged[]`, in Jevlin's order, cut at max_results.

    The order is the gateway's: pages returned by more providers first, then
    best per-provider rank, then provider order. The excerpt is the longest
    snippet any provider gave for the page; when only `excerpts` (quoted
    evidence) exist they are joined.
    """
    hits: list[dict[str, str]] = []
    seen: set[str] = set()
    for page in (payload or {}).get("merged") or []:
        if not isinstance(page, dict):
            continue
        url = str(page.get("url") or "")
        if not url or url in seen:
            continue
        seen.add(url)
        snippet = page.get("excerpt") or "\n".join(
            part for part in (page.get("excerpts") or []) if isinstance(part, str)
        )
        hits.append(_hit(url, page.get("title"), snippet))
        if len(hits) >= max_results:
            break
    return hits


def present_raw(payload: Any, *, max_results: int) -> list[dict[str, str]]:
    """Fan-out with the least processing that yields one list.

    Round-robin over the providers that answered, in the gateway's dispatch
    order: every provider's first result, then every provider's second, and
    so on. Duplicates are dropped only on the exact URL string the providers
    returned (each adapter has already canonicalised its own citations). No
    vote count, no re-ranking, no snippet choice: the first provider to show a
    page supplies its title and snippet.
    """
    lists: list[list[dict[str, Any]]] = []
    for cand in (payload or {}).get("candidates") or []:
        if not isinstance(cand, dict) or cand.get("status") != "ok":
            continue
        cits = [c for c in (cand.get("citations") or []) if isinstance(c, dict) and c.get("url")]
        if cits:
            lists.append(cits)
    hits: list[dict[str, str]] = []
    seen: set[str] = set()
    depth = max((len(cits) for cits in lists), default=0)
    for rank in range(depth):
        for cits in lists:
            if rank >= len(cits):
                continue
            cit = cits[rank]
            url = str(cit.get("url") or "")
            if not url or url in seen:
                continue
            seen.add(url)
            hits.append(_hit(url, cit.get("title"), cit.get("snippet")))
            if len(hits) >= max_results:
                return hits
    return hits


PRESENTATIONS = {"merged": present_merged, "raw": present_raw}


# ── metadata: what the harness's hop fields cannot carry ───────────────────


def arm_summary(payload: Any) -> list[dict[str, Any]]:
    """One row per dispatched arm: provider, status, results, cost, latency."""
    reasons: dict[str, str] = {}
    indexes = (payload or {}).get("indexes") or {}
    for row in indexes.get("by_provider") or []:
        if isinstance(row, dict) and row.get("provider") and row.get("reason"):
            reasons[str(row["provider"])] = str(row["reason"])
    arms: list[dict[str, Any]] = []
    for cand in (payload or {}).get("candidates") or []:
        if not isinstance(cand, dict):
            continue
        latency = cand.get("latency") if isinstance(cand.get("latency"), dict) else {}
        provider = str(cand.get("provider") or "")
        arms.append({
            "provider": provider,
            "kind": cand.get("kind"),
            "status": cand.get("status"),
            "reason": reasons.get(provider),
            "results": len(cand.get("citations") or []),
            "has_answer": bool(cand.get("answer")),
            "cost_micros": cand.get("cost_micros"),
            "latency_ms": latency.get("total_ms"),
            "error": cand.get("error"),
        })
    return arms


def jevlin_meta(payload: Any, hits: list[dict[str, str]], *, presentation: str) -> dict[str, Any]:
    decision = (payload or {}).get("decision") or {}
    usage = (payload or {}).get("usage") or {}
    arms = arm_summary(payload)
    merged = (payload or {}).get("merged")
    return {
        "jevlin_request_id": str((payload or {}).get("request_id") or ""),
        "jevlin_presentation": presentation,
        "jevlin_tier": decision.get("tier"),
        "jevlin_mode": decision.get("mode"),
        "jevlin_provider_set": decision.get("provider_set"),
        "jevlin_providers": list(decision.get("providers") or []),
        "jevlin_trimmed": list(decision.get("trimmed") or []),
        "jevlin_requested": bool(decision.get("requested")),
        "jevlin_router_version": decision.get("router_version"),
        "jevlin_ranker_version": decision.get("ranker_version"),
        "jevlin_cost_micros": usage.get("cost_micros"),
        "jevlin_latency_ms": usage.get("latency_ms"),
        "jevlin_cache_hit": bool(usage.get("cache_hit")),
        "jevlin_deadline_hit": bool(usage.get("deadline_hit")),
        "jevlin_pending": int(usage.get("pending") or 0),
        "jevlin_no_usable_result": bool((payload or {}).get("no_usable_result")),
        "jevlin_chosen": (payload or {}).get("chosen"),
        "jevlin_upstream_calls": sum(1 for arm in arms if arm["status"] not in NOT_CALLED),
        "jevlin_arms_ok": sum(1 for arm in arms if arm["status"] == "ok"),
        "jevlin_arms": arms,
        "jevlin_merged_total": len(merged) if isinstance(merged, list) else None,
        "jevlin_hits_shown": len(hits),
    }


# ── HTTP arms ──────────────────────────────────────────────────────────────


class JevlinSearch:
    """One Jevlin search as a harness backend. Subclasses set the arm."""

    name = "jevlin_balanced"
    tier: str | None = "balanced"
    presentation = "merged"
    pin_providers = False
    last_meta: dict[str, Any] | None = None

    def _key(self) -> str:
        key = (os.environ.get("JEVLIN_API_KEY") or "").strip()
        if not key:
            raise RuntimeError(f"JEVLIN_API_KEY is required for {self.name}")
        return key

    def providers(self) -> list[str]:
        if not self.pin_providers:
            return []
        names = pinned_providers()
        if not names:
            raise RuntimeError(
                f"JEVLIN_PROVIDERS must name the pinned provider set for {self.name}"
            )
        return names

    def request_body(self, query: str, max_results: int) -> dict[str, Any]:
        # view:"merged" adds `merged` and `indexes` and changes nothing else in
        # the response, so every arm asks for it and the raw presentation
        # reads `candidates` from the same body.
        body: dict[str, Any] = {"query": query, "max_results": int(max_results), "view": "merged"}
        if self.tier:
            body["tier"] = self.tier
        providers = self.providers()
        if providers:
            body["providers"] = providers
        return body

    def search(self, query: str, *, max_results: int | None = None) -> list[dict[str, str]]:
        harness = _harness()
        n = int(max_results or harness.DEFAULT_MAX_RESULTS)
        payload, meta = harness.vendor_call(
            self,
            "POST",
            api_url(),
            headers={"Authorization": f"Bearer {self._key()}", "Content-Type": "application/json"},
            json_body=self.request_body(query, n),
            timeout=timeout_s(),
        )
        hits = PRESENTATIONS[self.presentation](payload, max_results=n)
        self.last_meta = {
            **harness._search_meta(meta, hits),
            **jevlin_meta(payload, hits, presentation=self.presentation),
        }
        return hits

    def fetch(self, url: str, *, objective: str = "") -> dict[str, str]:
        del url, objective
        raise RuntimeError(f"{self.name} is search-only; Jevlin has no fetch endpoint")


class JevlinFast(JevlinSearch):
    name = "jevlin_fast"
    tier = "fast"


class JevlinBalanced(JevlinSearch):
    name = "jevlin_balanced"
    tier = "balanced"


class JevlinMax(JevlinSearch):
    name = "jevlin_max"
    tier = "max"


class JevlinSingle(JevlinSearch):
    """One provider through Jevlin: the plumbing without the fan-out.

    Compared with the same provider's direct runner this isolates what
    Jevlin's gateway adds or costs on its own: authentication, citation
    canonicalisation, metering, the 30 s deadline.
    """

    name = "jevlin_single"
    tier = None
    presentation = "raw"

    def providers(self) -> list[str]:
        return [(os.environ.get("JEVLIN_SINGLE_PROVIDER") or "exa").strip()]


class JevlinFanoutRaw(JevlinSearch):
    name = "jevlin_fanout_raw"
    tier = "balanced"
    presentation = "raw"
    pin_providers = True


class JevlinFanoutMerged(JevlinSearch):
    name = "jevlin_fanout_merged"
    tier = "balanced"
    presentation = "merged"
    pin_providers = True


# ── the CLI arm: production Jevlin as a coding agent receives it ───────────


class JevlinCli:
    """`jevlin search --stdin`, the customer-facing client.

    The CLI chooses no tier (the account's default applies), asks for no view,
    merges the router's candidates itself with its own URL rule, and reports
    `ok: true` whenever the protocol succeeded, even if every provider failed.
    All of that is what a coding agent gets in production, so none of it is
    corrected here. The model sees the client's merged list, top N.

    The CLI reads its router URL and key from its own config, not from this
    process's environment; `jevlin setup` / `jevlin login` on the run host
    decide where these searches go.
    """

    name = "jevlin_cli"
    last_meta: dict[str, Any] | None = None

    def config_path(self) -> str:
        # Its own variable, not JEVLIN_CONFIG: an operator's shell commonly exports
        # JEVLIN_CONFIG for their personal client, and python-dotenv never overrides
        # a variable the shell already set, so a JEVLIN_CONFIG line in .env would be
        # silently ignored and the arm would run on the operator's own config.
        return (os.environ.get("JEVLIN_CLI_CONFIG") or "").strip()

    def argv(self) -> list[str]:
        # -config: the bench config, passed explicitly so inherited environment
        # cannot redirect the arm. -no-flush: do not spawn the client's background
        # mining flush after the search; the bench config disables mining anyway,
        # this keeps stray processes off the run host. Neither changes the search.
        exe = (os.environ.get("JEVLIN_BIN") or "jevlin").strip()
        argv = [exe, "search"]
        if self.config_path():
            argv += ["-config", self.config_path()]
        return argv + ["--stdin", "-no-flush"]

    def env(self) -> dict[str, str]:
        env = dict(os.environ)
        if self.config_path():
            env["JEVLIN_CONFIG"] = self.config_path()
        return env

    def search(self, query: str, *, max_results: int | None = None) -> list[dict[str, str]]:
        harness = _harness()
        n = int(max_results or harness.DEFAULT_MAX_RESULTS)
        request = {"version": 1, "query": query, "max_results": n}
        argv = self.argv()
        audit = {"method": "CLI", "url": " ".join(argv), "headers": {}, "params": None, "json": request}
        started = time.perf_counter()
        try:
            proc = subprocess.run(
                argv,
                input=json.dumps(request, ensure_ascii=False),
                capture_output=True,
                text=True,
                timeout=timeout_s(),
                check=False,
                env=self.env(),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            self.last_meta = {
                "endpoint": audit["url"],
                "method": "CLI",
                "http_status": None,
                "http_ms": int((time.perf_counter() - started) * 1000),
                "response_bytes": 0,
                "vendor_request_id": "",
                "error": f"{type(exc).__name__}: {exc}",
                "request": audit,
                "response": None,
            }
            raise RuntimeError(f"{type(exc).__name__}: {exc}") from exc
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        try:
            envelope: Any = json.loads(proc.stdout) if proc.stdout.strip() else {}
        except ValueError:
            envelope = {"raw": proc.stdout[:500]}
        if not isinstance(envelope, dict):
            envelope = {"raw": str(envelope)[:500]}
        meta: dict[str, Any] = {
            "endpoint": audit["url"],
            "method": "CLI",
            "http_status": envelope.get("http_status"),
            "http_ms": elapsed_ms,
            "response_bytes": len((proc.stdout or "").encode("utf-8")),
            "vendor_request_id": str(envelope.get("request_id") or ""),
            "exit_code": proc.returncode,
            "request": audit,
            "response": {
                "exit_code": proc.returncode,
                "body": harness.clip_raw(envelope),
                "stderr": (proc.stderr or "")[:2000],
            },
        }
        if envelope.get("ok") is not True:
            err = envelope.get("error") if isinstance(envelope.get("error"), dict) else {}
            code = str(err.get("code") or envelope.get("code") or envelope.get("status") or "")
            message = str(err.get("message") or envelope.get("message") or (proc.stderr or "").strip()[:300])
            meta["error_body"] = json.dumps(envelope, default=str)[:800]
            self.last_meta = meta
            raise RuntimeError(f"jevlin search --stdin exit {proc.returncode}: {code} {message}".strip())
        result = envelope.get("result") if isinstance(envelope.get("result"), dict) else {}
        merged = result.get("merged") if isinstance(result.get("merged"), list) else []
        hits: list[dict[str, str]] = []
        seen: set[str] = set()
        for page in merged:
            if not isinstance(page, dict):
                continue
            url = str(page.get("url") or "")
            if not url or url in seen:
                continue
            seen.add(url)
            hits.append(_hit(url, page.get("title"), page.get("snippet")))
            if len(hits) >= n:
                break
        candidates = result.get("candidates") if isinstance(result.get("candidates"), list) else None
        decision = result.get("decision") if isinstance(result.get("decision"), dict) else {}
        usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
        arms = [
            {
                "provider": c.get("provider"),
                "kind": c.get("kind"),
                "status": c.get("status"),
                "results": len(c.get("citations") or []),
                "cost_micros": c.get("cost_micros"),
                "latency_ms": c.get("latency_ms"),
                "error": c.get("error"),
            }
            for c in (candidates or [])
            if isinstance(c, dict)
        ]
        self.last_meta = {
            **harness._search_meta(meta, hits),
            "jevlin_request_id": str(result.get("request_id") or envelope.get("request_id") or ""),
            "jevlin_presentation": "cli",
            "jevlin_client_merge": True,
            "jevlin_tier": decision.get("tier"),
            "jevlin_providers": list(decision.get("providers") or []),
            "jevlin_trimmed": list(decision.get("trimmed") or []),
            "jevlin_cost_micros": usage.get("cost_micros"),
            "jevlin_latency_ms": usage.get("latency_ms") or result.get("latency_ms"),
            "jevlin_cache_hit": bool(usage.get("cache_hit")),
            "jevlin_pending": int(usage.get("pending") or 0),
            # The envelope has no no_usable_result flag; infer it when the
            # candidates are present, as they are in the CLI's default view.
            "jevlin_no_usable_result": bool(
                not merged and arms and all(arm["status"] != "ok" for arm in arms)
            ),
            "jevlin_upstream_calls": sum(1 for arm in arms if arm["status"] not in NOT_CALLED) if arms else None,
            "jevlin_arms_ok": sum(1 for arm in arms if arm["status"] == "ok") if arms else None,
            "jevlin_arms": arms,
            "jevlin_merged_total": len(merged),
            "jevlin_hits_shown": len(hits),
            "jevlin_mining": envelope.get("mining"),
        }
        return hits

    def fetch(self, url: str, *, objective: str = "") -> dict[str, str]:
        del url, objective
        raise RuntimeError("jevlin_cli is search-only; the jevlin CLI has no fetch")


SEARCH_ONLY: tuple[str, ...] = (
    "jevlin_fast",
    "jevlin_balanced",
    "jevlin_max",
    "jevlin_single",
    "jevlin_fanout_raw",
    "jevlin_fanout_merged",
    "jevlin_cli",
)

BACKENDS = {
    "jevlin_fast": JevlinFast,
    "jevlin_balanced": JevlinBalanced,
    "jevlin_max": JevlinMax,
    "jevlin_single": JevlinSingle,
    "jevlin_fanout_raw": JevlinFanoutRaw,
    "jevlin_fanout_merged": JevlinFanoutMerged,
    "jevlin_cli": JevlinCli,
}
