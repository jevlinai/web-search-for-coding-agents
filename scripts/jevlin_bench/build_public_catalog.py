#!/usr/bin/env python3
"""Turn the public OB-Code-Websearch rows into the ticket tree the harness loads.

The public dataset (openbenchmarks/OB-Code-Websearch, data/eval.jsonl) carries
task_id, vendor, ticket, starter_code and gold_code. The harness needs more
than that per ticket: the gold tokens the judge checks (`required`), the
tokens whose presence in a query is flagged as a leak (`query_forbidden`), the
output filename, and a prompt that includes the file to patch. The private
catalog has all of that; the public rows do not. This script derives it.

Derivation of `required`: identifier-like tokens (4+ chars) on gold lines that
differ from the starter, excluding comment lines and anything that already
occurs as a substring of the starter. That is a reconstruction, not the
maintainers' catalog, so a ticket's pass/fail here is not comparable to the
board. The harness's selftest proves two things mechanically for every
ticket: the gold file passes and the starter fails. A ticket that cannot
satisfy both is written with kind "unscorable" and excluded by default.

Usage:
  python3 scripts/jevlin_bench/build_public_catalog.py --download --out datasets
  python3 scripts/jevlin_bench/build_public_catalog.py --src eval.jsonl --out datasets
Then:
  DATASETS_ROOT=datasets PYTHONPATH=scripts python -m coding_search selftest
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import urllib.request
from pathlib import Path

HF_URL = "https://huggingface.co/datasets/openbenchmarks/OB-Code-Websearch/resolve/main/data/eval.jsonl"
OUTFILE = "file.py"
TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}")
SOURCE_URL = re.compile(r"https?://[^\s#]+", re.I)
# Tokens that are Python or httpx vocabulary rather than evidence from the docs.
STOP = {
    "import", "return", "response", "headers", "json", "httpx", "environ", "True", "False", "None",
    "self", "dict", "list", "print", "raise", "except", "with", "from", "def", "async", "await",
    "Authorization", "Bearer", "Content", "Type", "application", "Accept", "params", "data",
    "timeout", "status", "raise_for_status", "base_url", "BASE", "TOKEN", "post", "get", "put", "patch", "delete",
}

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def gold_only_lines(starter: str, gold: str) -> list[str]:
    out: list[str] = []
    for line in difflib.unified_diff(starter.splitlines(), gold.splitlines(), lineterm="", n=0):
        if line.startswith("+") and not line.startswith("+++"):
            out.append(line[1:])
    return out


QUOTED = re.compile(r"""(?P<q>["'])(?P<s>(?:(?!(?P=q)).)*)(?P=q)""")
JSON_LITERALS = {"true", "false", "null", "True", "False", "None"}


def evidence_tokens(line: str) -> list[str]:
    """Tokens on one gold line that could only have come from the docs.

    Keeps: identifiers outside strings, every token of a string that is a path
    (contains '/'), and the token of a string in key position (followed by
    ':'). Drops: tokens of a string in value position (preceded by ':'), which
    are sample data — "PetShop", "Engineering", "Quarterly report" — that the
    maintainers' catalog would not demand verbatim.
    """
    kept: list[str] = []
    cursor = 0
    for match in QUOTED.finditer(line):
        for token in TOKEN.findall(line[cursor:match.start()]):
            kept.append(token)
        text = match.group("s")
        before = line[:match.start()].rstrip()
        after = line[match.end():].lstrip()
        is_path = "/" in text
        is_key = after.startswith(":")
        is_value = before.endswith(":") or before.endswith("=") and not is_path
        if is_path or is_key or not is_value:
            kept.extend(TOKEN.findall(text))
        cursor = match.end()
    kept.extend(TOKEN.findall(line[cursor:]))
    return kept


def short_segment_fragments(line: str, starter: str) -> list[str]:
    """Path fragments around new segments too short to be needles on their own.

    The judge drops needles under four characters, so a ticket whose whole
    change is `/api/now/v1/attachment` (new segment `v1`) would be unscorable.
    The needle check is a plain substring test, so `now/v1/attachment` works:
    each new short segment is emitted with its neighbours.
    """
    out: list[str] = []
    for match in QUOTED.finditer(line):
        text = match.group("s")
        if "/" not in text:
            continue
        segments = [seg for seg in text.split("/") if seg]
        for index, seg in enumerate(segments):
            if "{" in seg or len(seg) >= 4 or seg in starter.split("/") or not seg.isalnum():
                continue
            if seg in starter and f"/{seg}/" in starter:
                continue
            left = segments[index - 1] if index > 0 and "{" not in segments[index - 1] else ""
            right = segments[index + 1] if index + 1 < len(segments) and "{" not in segments[index + 1] else ""
            fragment = "/".join(part for part in (left, seg, right) if part)
            if len(fragment) >= 4 and fragment not in starter and fragment not in out:
                out.append(fragment)
    return out


def derive_required(starter: str, gold: str) -> list[str]:
    found: list[str] = []
    for line in gold_only_lines(starter, gold):
        if line.lstrip().startswith("#"):
            continue
        for token in evidence_tokens(line):
            if token in STOP or token in JSON_LITERALS or token.isdigit() or token in found:
                continue
            # Substring semantics match the judge, which checks `needle in text`.
            if token in starter:
                continue
            found.append(token)
        for fragment in short_segment_fragments(line, starter):
            if fragment not in found:
                found.append(fragment)
    return found


def source_urls(gold: str) -> list[str]:
    urls: list[str] = []
    for line in gold.splitlines():
        if line.lstrip().startswith("#"):
            for match in SOURCE_URL.findall(line):
                url = match.rstrip(").,;\"'")
                if url not in urls:
                    urls.append(url)
    return urls


def prompt_text(ticket: str, starter: str) -> str:
    return (
        f"{ticket.strip()}\n\n"
        f"Current `{OUTFILE}`:\n\n```python\n{starter.rstrip()}\n```\n"
    )


CORRECTIONS = ROOT / "experiments" / "catalog-corrections.json"


def load_corrections() -> dict:
    """Reviewed corrections to the derived catalog, committed beside the experiments.

    `recase`: {task_id: {gold_token: documented_token}} — the public gold's casing differs from
    the documentation it cites; applied to the catalog AND to the gold file, so the gold still
    passes the judge. `drop`: {task_id: [token, ...]} — sample data the derivation kept that
    the agent could not have known. Every entry carries a `why`. Nothing here is tuned to a
    result; the file is frozen with the catalog.
    """
    if not CORRECTIONS.is_file():
        return {"recase": {}, "drop": {}}
    data = json.loads(CORRECTIONS.read_text())
    return {"recase": data.get("recase") or {}, "drop": data.get("drop") or {}}


def apply_corrections(tid: str, required: list[str], gold: str, corrections: dict) -> tuple[list[str], str, list[str]]:
    applied: list[str] = []
    recase = corrections["recase"].get(tid) or {}
    for old, entry in recase.items():
        new = entry["to"] if isinstance(entry, dict) else str(entry)
        if old in required:
            required = [new if token == old else token for token in required]
            gold = gold.replace(old, new)
            applied.append(f"recase {old}->{new}")
    for entry in corrections["drop"].get(tid) or []:
        token = entry["token"] if isinstance(entry, dict) else str(entry)
        if token in required:
            required = [t for t in required if t != token]
            applied.append(f"drop {token}")
    return required, gold, applied


def read_rows(src: Path | None, download: bool) -> list[dict]:
    if download:
        with urllib.request.urlopen(HF_URL, timeout=60) as resp:  # noqa: S310
            text = resp.read().decode("utf-8")
    elif src is not None:
        text = src.read_text()
    else:
        raise SystemExit("pass --src eval.jsonl or --download")
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    for key in ("task_id", "vendor", "ticket", "starter_code", "gold_code"):
        missing = [r.get("task_id", "?") for r in rows if key not in r]
        if missing:
            raise SystemExit(f"rows missing {key}: {missing[:5]}")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=None, help="local eval.jsonl")
    parser.add_argument("--download", action="store_true", help=f"fetch {HF_URL}")
    parser.add_argument("--out", type=Path, default=ROOT / "datasets", help="DATASETS_ROOT to write")
    parser.add_argument("--include-unscorable", action="store_true", help="keep tickets whose derived tokens fail the gold/starter check")
    args = parser.parse_args(argv)

    from coding_search.judge import score_needles  # noqa: E402
    from coding_search.tasks import CodingTask  # noqa: E402

    rows = read_rows(args.src, args.download)
    corrections = load_corrections()
    tasks_dir = args.out / "web-search" / "coding" / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict] = []
    derivation: list[dict] = []
    unscorable: list[str] = []
    for row in rows:
        tid = row["task_id"]
        starter, gold = row["starter_code"], row["gold_code"]
        required = derive_required(starter, gold)
        required, gold, applied = apply_corrections(tid, required, gold, corrections)
        task = CodingTask(id=tid, repo=row["vendor"], outfile=OUTFILE, prompt="", required=required,
                          query_forbidden=list(required))
        gold_score = score_needles(gold, task)
        starter_score = score_needles(starter, task)
        scorable = bool(required) and gold_score.passed and not starter_score.passed
        derivation.append({
            "task_id": tid,
            "vendor": row["vendor"],
            "required": required,
            "gold_passes": gold_score.passed,
            "gold_notes": gold_score.notes,
            "starter_fails": not starter_score.passed,
            "source_urls": source_urls(gold),
            "scorable": scorable,
            "corrections": applied,
        })
        if not scorable:
            unscorable.append(tid)
            if not args.include_unscorable:
                continue
        dest = tasks_dir / tid
        (dest / "starter").mkdir(parents=True, exist_ok=True)
        (dest / "gold").mkdir(parents=True, exist_ok=True)
        (dest / "prompt.txt").write_text(prompt_text(row["ticket"], starter))
        (dest / "starter" / OUTFILE).write_text(starter if starter.endswith("\n") else starter + "\n")
        (dest / "gold" / OUTFILE).write_text(gold if gold.endswith("\n") else gold + "\n")
        items.append({
            "id": tid,
            "repo": row["vendor"],
            "required": required,
            "query_forbidden": list(required),
            "docs": source_urls(gold),
            "hops": [],
            **({"kind": "unscorable"} if not scorable else {}),
        })
    catalog = {
        "source": {"dataset": "openbenchmarks/OB-Code-Websearch", "file": "data/eval.jsonl", "derived_by": "scripts/jevlin_bench/build_public_catalog.py",
                   "corrections": str(CORRECTIONS.relative_to(ROOT)) if CORRECTIONS.is_file() else None},
        "defaults": {"outfile": OUTFILE, "cite_urls": True, "max_turns": 32, "max_searches": 5, "max_fetches": 5, "kind": "patch"},
        "items": items,
    }
    (tasks_dir / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    (tasks_dir / "derivation.json").write_text(json.dumps(derivation, indent=2) + "\n")
    print(f"rows={len(rows)} written={len(items)} unscorable={len(unscorable)} -> {tasks_dir}")
    for entry in derivation:
        flag = "" if entry["scorable"] else "  UNSCORABLE"
        fixes = f"  [{'; '.join(entry['corrections'])}]" if entry["corrections"] else ""
        print(f"  {entry['task_id']}: {entry['required']}{flag}{fixes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
