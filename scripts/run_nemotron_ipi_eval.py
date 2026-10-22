#!/usr/bin/env python3
"""none vs frozen eval on NVIDIA's agentic indirect-prompt-injection set.

The agent sees the dataset's domain prompt (both sides). ``frozen`` also
prepends the frozen harness. The injection is not in the user message; it is
returned by the simulated read tool. ASR counts a real ``target_tool`` call
with the target arguments. Model errors stay errors.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from contract_heldout_ipi.agent import (  # noqa: E402
    ExperimentModelConfig,
    ModelAPIError,
    ModelClientError,
    build_role_client,
)
from contract_heldout_ipi.dojo.transfer import (  # noqa: E402
    DEFAULT_FROZEN_HARNESS,
    load_frozen_system_prompt,
)
from contract_heldout_ipi.eval.hf_agent import build_system_prompt  # noqa: E402
from contract_heldout_ipi.eval.nemotron_ipi import (  # noqa: E402
    DEFAULT_NEMOTRON_PATH,
    load_nemotron_ipi,
    run_nemotron_episode,
)

DEFENSES = ("none", "frozen")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--defense",
        action="append",
        choices=DEFENSES,
        dest="defenses",
    )
    parser.add_argument("--max-examples", type=int, default=25)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--domain", default=None)
    parser.add_argument("--data", type=Path, default=DEFAULT_NEMOTRON_PATH)
    parser.add_argument("--harness", type=Path, default=DEFAULT_FROZEN_HARNESS)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--list", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list:
        examples = load_nemotron_ipi(path=args.data, domain=args.domain)
        domains: dict[str, int] = {}
        for example in examples:
            domains[example.domain] = domains.get(example.domain, 0) + 1
        print(f"nemotron_ipi: ready, {len(examples)} examples")
        for domain, count in sorted(domains.items()):
            print(f"  {domain}: {count}")
        return 0
    if args.config is None or args.output is None:
        print(
            "Configuration error: --config and --output are required unless --list is used",
            file=sys.stderr,
        )
        return 2

    defenses = tuple(args.defenses) if args.defenses else DEFENSES
    max_examples = None if args.all else args.max_examples
    examples = load_nemotron_ipi(
        path=args.data, max_examples=max_examples, domain=args.domain
    )
    frozen_prompt = load_frozen_system_prompt(args.harness)
    config = ExperimentModelConfig.from_file(args.config)
    client = build_role_client(
        config,
        "task_agent",
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
    )
    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "nemotron_ipi",
        "dataset": "nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1",
        "config": str(args.config),
        "harness": str(args.harness),
        "model": config.for_role("task_agent").model,
        "max_examples": max_examples,
        "domain": args.domain,
        "max_steps": args.max_steps,
        "defenses": list(defenses),
        "cases": [],
    }
    done: set[tuple[str, str]] = set()
    if args.resume and args.output.is_file():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        report["created_at"] = existing.get("created_at", report["created_at"])
        report["resumed_at"] = datetime.now(timezone.utc).isoformat()
        report["cases"] = list(existing.get("cases") or [])
        done = {
            (str(case.get("id")), str(case.get("defense")))
            for case in report["cases"]
            if isinstance(case, dict)
        }
        print(f"resume: kept {len(done)} cases", flush=True)

    print(f"nemotron_ipi: {len(examples)} examples", flush=True)
    for defense in defenses:
        for index, example in enumerate(examples, start=1):
            if (example.example_id, defense) in done:
                continue
            system = build_system_prompt(
                defense=defense,
                frozen_prompt=frozen_prompt,
                extra_system=example.system_text,
            )
            case: dict[str, Any] = {
                "dataset": "nemotron_ipi",
                "id": example.example_id,
                "domain": example.domain,
                "defense": defense,
                "attack_category": example.attack_category,
                "target_tool": example.target_tool,
            }
            try:
                episode = run_nemotron_episode(
                    example, client, system, max_steps=args.max_steps
                )
                case.update(
                    {
                        "attack_success": episode.attack_success,
                        "passed": not episode.attack_success,
                        "reason": episode.reason,
                        "saw_injection": episode.saw_injection,
                        "content": (episode.content or "")[:500] or None,
                        "tool_calls": list(episode.tool_calls),
                    }
                )
            except (ModelAPIError, ModelClientError, TypeError, ValueError) as exc:
                case.update(
                    {
                        "attack_success": False,
                        "passed": False,
                        "error": str(exc),
                        "reason": "model error",
                        "saw_injection": False,
                    }
                )
            report["cases"].append(case)
            mark = "ASR" if case.get("attack_success") else "BLOCK"
            print(
                f"  [{defense}] {index}/{len(examples)} {example.example_id} "
                f"{mark} {case.get('error') or case.get('reason')}",
                flush=True,
            )
            _write_report(args.output, report, defenses)

    _print_summary(report)
    print(f"wrote {args.output}", flush=True)
    return 0


def _write_report(
    path: Path, report: dict[str, Any], defenses: Sequence[str]
) -> None:
    cells: dict[str, Any] = {}
    for defense in defenses:
        rows = [case for case in report["cases"] if case.get("defense") == defense]
        if rows:
            cells[f"nemotron_ipi/{defense}"] = _cell(rows)
        domains = sorted({str(case.get("domain")) for case in rows})
        for domain in domains:
            domain_rows = [case for case in rows if case.get("domain") == domain]
            cells[f"nemotron_ipi/{domain}/{defense}"] = _cell(domain_rows)
    report["cells"] = cells
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _cell(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    attacks = sum(1 for row in rows if row.get("attack_success"))
    errors = sum(1 for row in rows if row.get("error"))
    seen = sum(1 for row in rows if row.get("saw_injection"))
    scored = len(rows) - errors
    return {
        "n": len(rows),
        "errors": errors,
        "scored": scored,
        "attack_success": attacks,
        "attack_success_rate": (attacks / scored) if scored else None,
        "saw_injection": seen,
    }


def _print_summary(report: Mapping[str, Any]) -> None:
    print("\n=== nemotron IPI none vs frozen ===", flush=True)
    for key, stats in report.get("cells", {}).items():
        if key.count("/") != 1:
            continue
        rate = stats.get("attack_success_rate")
        rate_s = "n/a" if rate is None else f"{rate:.1%}"
        print(
            f"{key}: ASR {stats['attack_success']}/{stats['scored']}={rate_s} "
            f"saw_injection={stats['saw_injection']} errors={stats['errors']}",
            flush=True,
        )


if __name__ == "__main__":
    raise SystemExit(main())
