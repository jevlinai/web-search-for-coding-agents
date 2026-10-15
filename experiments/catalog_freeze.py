#!/usr/bin/env python3
"""Freeze or verify the public ticket tree.

  freeze   hash every file under datasets/web-search/coding/tasks (sorted relative paths) and
           write experiments/FROZEN.sha256: one tree hash plus the per-file list
  verify   recompute and compare; exit 1 on any difference (pilot.sh runs this first)

The tree is regenerated from the public rows + experiments/catalog-corrections.json by
build_public_catalog.py, so the hash pins the exact inputs every pass was judged against.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "datasets" / "web-search" / "coding" / "tasks"
FROZEN = ROOT / "experiments" / "FROZEN.sha256"
SKIP = {"review.json"}  # review output is evidence, not an input


def file_hashes() -> list[tuple[str, str]]:
    out = []
    for path in sorted(TASKS.rglob("*")):
        if path.is_file() and path.name not in SKIP:
            out.append((str(path.relative_to(TASKS)), hashlib.sha256(path.read_bytes()).hexdigest()))
    return out


def tree_hash(entries: list[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for rel, h in entries:
        digest.update(f"{h}  {rel}\n".encode())
    return digest.hexdigest()


def main(argv: list[str]) -> int:
    if not TASKS.is_dir():
        print(f"no ticket tree at {TASKS}; run build_public_catalog.py first", file=sys.stderr)
        return 2
    entries = file_hashes()
    tree = tree_hash(entries)
    cmd = argv[1] if len(argv) > 1 else "verify"
    if cmd == "freeze":
        lines = [f"tree {tree}", f"files {len(entries)}"] + [f"{h}  {rel}" for rel, h in entries]
        FROZEN.write_text("\n".join(lines) + "\n")
        print(f"frozen: tree {tree} files={len(entries)} -> {FROZEN}")
        return 0
    if cmd == "verify":
        if not FROZEN.is_file():
            print("not frozen: experiments/FROZEN.sha256 missing", file=sys.stderr)
            return 1
        expected = FROZEN.read_text().splitlines()[0].split()[1]
        if expected != tree:
            print(f"CATALOG MISMATCH: frozen {expected} current {tree}", file=sys.stderr)
            frozen_files = {line.split("  ", 1)[1]: line.split()[0] for line in FROZEN.read_text().splitlines()[2:]}
            current = dict((rel, h) for rel, h in entries)
            for rel in sorted(set(frozen_files) | set(current)):
                if frozen_files.get(rel) != current.get(rel):
                    print(f"  differs: {rel}", file=sys.stderr)
            return 1
        print(f"catalog verified: tree {tree} files={len(entries)}")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
