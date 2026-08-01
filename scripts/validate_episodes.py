#!/usr/bin/env python3
"""Validate all episode contracts with schema + referee checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from contract_heldout_ipi.contract.loader import load_episode
from contract_heldout_ipi.referee import Referee


def main() -> int:
    episode_dirs = [ROOT / "episodes" / "train", ROOT / "episodes" / "heldout"]
    referee = Referee()
    reports = []

    for directory in episode_dirs:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.json")):
            raw = json.loads(path.read_text(encoding="utf-8"))
            episode = load_episode(path, check_schema=False)
            report = referee.validate(episode, raw=raw)
            reports.append(report)
            print(report.summary())

    accepted = sum(1 for r in reports if r.ok)
    rejected = len(reports) - accepted
    print(f"\nTotal: {len(reports)} | accept: {accepted} | reject: {rejected}")
    return 0 if rejected == 0 and reports else 1


if __name__ == "__main__":
    raise SystemExit(main())
