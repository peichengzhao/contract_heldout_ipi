#!/usr/bin/env python3
"""IterInject closed loop on AgentDojo.

``--defense none`` searches an undefended victim. ``--defense frozen`` searches
with the frozen harness on the victim only; the optimizer still sees traces,
not the harness text. Replay scores saved undefended payloads once.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from contract_heldout_ipi.agent import (  # noqa: E402
    ExperimentModelConfig,
    ModelClientError,
    ModelConfigurationError,
    build_role_client,
)
from contract_heldout_ipi.dojo.adapter import (  # noqa: E402
    AgentDojoDependencyError,
    load_suite,
)
from contract_heldout_ipi.dojo.iterinject_runner import (  # noqa: E402
    make_victim,
    pair_context,
)
from contract_heldout_ipi.dojo.pipeline import build_dojo_pipeline  # noqa: E402
from contract_heldout_ipi.dojo.transfer import (  # noqa: E402
    DEFAULT_FROZEN_HARNESS,
    load_frozen_system_prompt,
    load_transfer_split,
    default_transfer_split,
)
from contract_heldout_ipi.eval.iterinject import (  # noqa: E402
    CampaignLog,
    LoopBudget,
    Seed,
    assert_resume_search,
    fresh_seed_bank,
    replay_payloads_from_search,
    run_target,
    search_victim_setup,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--phase", choices=("search", "replay"), default="search")
    parser.add_argument("--suite", default="banking")
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--split", default=None)
    parser.add_argument("--user-task", action="append", default=[])
    parser.add_argument("--injection-task", action="append", default=[])
    parser.add_argument("--max-pairs", type=int, default=1)
    parser.add_argument("--all-pairs", action="store_true")
    parser.add_argument("--max-iterations", type=int, default=7)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--patience-extend", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--max-seeds", type=int)
    parser.add_argument("--max-synthesized", type=int, default=1)
    parser.add_argument("--optimizer-role", default="attack")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--optimizer-max-tokens", type=int, default=1024)
    parser.add_argument("--optimizer-temperature", type=float, default=0.7)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--search-output", type=Path)
    parser.add_argument("--defense", choices=("none", "frozen"), default="none")
    parser.add_argument("--harness", type=Path, default=DEFAULT_FROZEN_HARNESS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.phase == "search":
            return _search(args)
        return _replay(args)
    except (ValueError, ModelConfigurationError, ModelClientError) as exc:
        print(f"IterInject error: {exc}", file=sys.stderr)
        return 2
    except AgentDojoDependencyError as exc:
        print(str(exc), file=sys.stderr)
        return 2


def _search(args: argparse.Namespace) -> int:
    if args.config is None or args.output is None:
        raise ValueError("search requires --config and --output")
    if args.optimizer_role != "attack":
        raise ValueError("optimizer role must be attack")
    config = ExperimentModelConfig.from_file(args.config)
    _reject_gpt(config.attack_model)
    budget = LoopBudget(
        max_iterations=args.max_iterations,
        patience=args.patience,
        patience_extend=args.patience_extend,
        batch_size=args.batch_size,
        max_seeds=args.max_seeds,
        max_synthesized=args.max_synthesized,
    )
    split = _load_split(args)
    suite = load_suite(split["benchmark_version"], split["suite"])
    pairs = _select_pairs(split, args)
    done: set[str] = set()
    bank = fresh_seed_bank()
    finished: list[dict[str, Any]] = []
    harness_text = None
    if args.defense == "frozen":
        harness_text = load_frozen_system_prompt(args.harness)
    search_defense, victim_system_message = search_victim_setup(
        args.defense, harness_text
    )
    if args.resume and args.output.is_file():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        assert_resume_search(
            previous, search_defense=search_defense, suite=split["suite"]
        )
        bank = [Seed.from_dict(item) for item in previous.get("seed_bank", [])]
        if not bank:
            bank = fresh_seed_bank()
        for pair in previous.get("pairs", []):
            if isinstance(pair, dict) and pair.get("done"):
                finished.append(pair)
                done.add(_pair_key(pair["user_task_id"], pair["injection_task_id"]))
        print(f"resume: {len(done)} pairs already done", flush=True)

    pipeline = build_dojo_pipeline(
        config,
        defense="none",
        temperature=0.0,
        timeout=args.timeout,
        max_retries=args.max_retries,
        max_tokens=args.max_tokens,
        system_message=victim_system_message,
    )
    optimizer = build_role_client(
        config,
        "attack",
        timeout=args.timeout,
        temperature=args.optimizer_temperature,
        max_tokens=args.optimizer_max_tokens,
        max_retries=args.max_retries,
    )
    log = CampaignLog(
        payloads=[
            str(pair["best_payload"])
            for pair in finished
            if pair.get("success") and isinstance(pair.get("best_payload"), str)
        ]
    )
    report = _search_report(
        args,
        config.attack_model,
        config.task_agent_model or config.attack_model,
        split,
        budget,
        search_defense=search_defense,
    )
    report["pairs"] = finished
    report["seed_bank"] = [seed.to_dict() for seed in bank]

    pending = [pair for pair in pairs if _pair_key(*pair) not in done]
    print(
        f"search: {len(pending)} pairs on {split['suite']} defense={search_defense} "
        f"harness_visible_to_optimizer=false optimizer={config.attack_model}",
        flush=True,
    )
    for index, (user_id, injection_id) in enumerate(pending, start=1):
        user_task = suite.get_user_task_by_id(user_id)
        injection_task = suite.get_injection_task_by_id(injection_id)
        context = pair_context(suite, user_task, injection_task)
        print(
            f"[search] {index}/{len(pending)} {user_id} x {injection_id} "
            f"vectors={list(context.vectors)} tools={list(context.attacker_tools)}",
            flush=True,
        )
        if not context.vectors:
            finished.append(
                {
                    "user_task_id": user_id,
                    "injection_task_id": injection_id,
                    "goal": context.goal,
                    "done": True,
                    "success": False,
                    "best_status": "error",
                    "best_payload": None,
                    "error": "no visible injection vector",
                }
            )
            _write(args.output, report, bank, finished)
            continue
        victim = make_victim(
            suite, pipeline, user_task, injection_task, context.vectors
        )

        def _optimize(prompt: str) -> str:
            response = optimizer.complete(
                [{"role": "user", "content": prompt}],
                [],
            )
            return response.content or ""

        def _synthesize(prompt: str) -> str:
            response = optimizer.complete(
                [{"role": "user", "content": prompt}],
                [],
            )
            return response.content or ""

        def _on_iteration(seed: Seed, step: Any) -> None:
            print(
                f"  {seed.name} iter {step.iteration}/{budget.max_iterations} "
                f"{step.status}",
                flush=True,
            )

        result = run_target(
            bank,
            goal=context.goal,
            attacker_tools=context.attacker_tools,
            victim=victim,
            optimize=_optimize,
            synthesize=_synthesize,
            budget=budget,
            cross_successes=log.payloads,
            on_iteration=_on_iteration,
        )
        if result.success:
            log.add(result.best_payload)
        record = {
            "user_task_id": user_id,
            "injection_task_id": injection_id,
            "goal": context.goal,
            "vectors": list(context.vectors),
            "attacker_tools": list(context.attacker_tools),
            "done": True,
            "success": result.success,
            "best_payload": result.best_payload,
            "best_status": result.best_status,
            "best_value": result.best_value,
            "best_utility": result.best_utility,
            "attack_success": result.attack_success,
            "victim_calls": result.victim_calls,
            "optimizer_calls": result.optimizer_calls,
            "synthesized": list(result.synthesized),
            "seed_runs": [run.to_dict() for run in result.seed_runs],
            "error": None,
        }
        finished.append(record)
        _write(args.output, report, bank, finished)
        print(
            f"  best {result.best_status} success={result.success} "
            f"calls={result.victim_calls}",
            flush=True,
        )
    _summarize(report)
    _write(args.output, report, bank, finished)
    _print_summary(report)
    print(f"wrote {args.output}", flush=True)
    return 0


def _replay(args: argparse.Namespace) -> int:
    if args.config is None or args.output is None or args.search_output is None:
        raise ValueError("replay requires --config, --search-output, and --output")
    search = json.loads(args.search_output.read_text(encoding="utf-8"))
    selected = replay_payloads_from_search(search)
    if not selected:
        raise ValueError("search report has no completed payloads to replay")
    config = ExperimentModelConfig.from_file(args.config)
    system_message = None
    pipeline_defense = "none"
    if args.defense == "frozen":
        system_message = load_frozen_system_prompt(args.harness)
    elif args.defense != "none":
        raise ValueError("replay defense must be none or frozen")
    pipeline = build_dojo_pipeline(
        config,
        defense=pipeline_defense,
        temperature=0.0,
        timeout=args.timeout,
        max_retries=args.max_retries,
        max_tokens=args.max_tokens,
        system_message=system_message,
    )
    suite = load_suite(search["benchmark_version"], search["suite"])
    cases: list[dict[str, Any]] = []
    print(
        f"replay: {len(selected)} payloads defense={args.defense} "
        f"optimizer_calls=0",
        flush=True,
    )
    for index, item in enumerate(selected, start=1):
        user_task = suite.get_user_task_by_id(item["user_task_id"])
        injection_task = suite.get_injection_task_by_id(item["injection_task_id"])
        context = pair_context(suite, user_task, injection_task)
        print(
            f"[replay] {index}/{len(selected)} {item['user_task_id']} x "
            f"{item['injection_task_id']}",
            flush=True,
        )
        if not context.vectors:
            cases.append({**item, "attack_success": False, "error": "no visible vector"})
            continue
        observation = make_victim(
            suite, pipeline, user_task, injection_task, context.vectors
        )(item["payload"])
        success = observation.attack_success is True
        print(
            f"  {'ASR' if success else 'BLOCK'} utility={observation.utility} "
            f"error={observation.error}",
            flush=True,
        )
        cases.append(
            {
                "user_task_id": item["user_task_id"],
                "injection_task_id": item["injection_task_id"],
                "attack_success": success,
                "utility": observation.utility,
                "tool_names": list(observation.tool_names),
                "error": observation.error,
            }
        )
    scored = [case for case in cases if case.get("error") is None]
    successes = sum(bool(case.get("attack_success")) for case in scored)
    report = {
        "schema_version": 1,
        "method": "iterinject",
        "phase": "replay",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "search_output": str(args.search_output),
        "defense": args.defense,
        "harness_visible_to_optimizer": False,
        "optimizer_calls": 0,
        "suite": search.get("suite"),
        "benchmark_version": search.get("benchmark_version"),
        "summary": {
            "n": len(selected),
            "scored": len(scored),
            "attack_success": successes,
            "errors": len(cases) - len(scored),
            "attack_success_rate": (successes / len(scored)) if scored else None,
        },
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    summary = report["summary"]
    rate = summary["attack_success_rate"]
    rate_s = "n/a" if rate is None else f"{rate:.1%}"
    print(
        f"replay {args.defense}: ASR {summary['attack_success']}/{summary['scored']}"
        f"={rate_s} errors={summary['errors']}",
        flush=True,
    )
    print(f"wrote {args.output}", flush=True)
    return 0


def _load_split(args: argparse.Namespace) -> dict[str, Any]:
    default_file, default_name = default_transfer_split(args.suite)
    path = args.split_file or default_file
    name = args.split or default_name
    loaded = load_transfer_split(path, name)
    loaded["split"] = name
    loaded["split_file"] = str(path)
    return loaded


def _select_pairs(
    split: dict[str, Any], args: argparse.Namespace
) -> list[tuple[str, str]]:
    users = list(args.user_task) or list(split["user_task_ids"])
    injections = list(args.injection_task) or list(split["injection_task_ids"])
    pairs = [(user, injection) for user in users for injection in injections]
    if not args.all_pairs:
        if args.max_pairs < 1:
            raise ValueError("max_pairs must be at least 1")
        pairs = pairs[: args.max_pairs]
    if not pairs:
        raise ValueError("no user/injection pairs selected")
    return pairs


def _search_report(
    args: argparse.Namespace,
    optimizer_model: str,
    task_model: str,
    split: dict[str, Any],
    budget: LoopBudget,
    *,
    search_defense: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "method": "iterinject",
        "phase": "search",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "search_defense": search_defense,
        "harness_path": (
            str(args.harness) if search_defense == "frozen" else None
        ),
        "harness_visible_to_optimizer": False,
        "suite": split["suite"],
        "benchmark_version": split["benchmark_version"],
        "split": split["split"],
        "task_model": task_model,
        "optimizer_model": optimizer_model,
        "budget": {
            "max_iterations": budget.max_iterations,
            "patience": budget.patience,
            "patience_extend": budget.patience_extend,
            "batch_size": budget.batch_size,
            "max_seeds": budget.max_seeds,
            "max_synthesized": budget.max_synthesized,
        },
        "pairs": [],
        "seed_bank": [],
        "summary": {},
    }


def _write(
    path: Path,
    report: dict[str, Any],
    bank: Sequence[Seed],
    pairs: list[dict[str, Any]],
) -> None:
    report["pairs"] = pairs
    report["seed_bank"] = [seed.to_dict() for seed in bank]
    _summarize(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _summarize(report: dict[str, Any]) -> None:
    pairs = [pair for pair in report.get("pairs", []) if pair.get("done")]
    scored = [
        pair
        for pair in pairs
        if pair.get("best_status") not in {None, "error"} or pair.get("success")
    ]
    successes = sum(bool(pair.get("success")) for pair in scored)
    errors = len(pairs) - len(scored)
    report["summary"] = {
        "pairs": len(pairs),
        "scored": len(scored),
        "attack_success": successes,
        "errors": errors,
        "attack_success_rate": (successes / len(scored)) if scored else None,
        "victim_calls": sum(int(pair.get("victim_calls") or 0) for pair in pairs),
        "optimizer_calls": sum(int(pair.get("optimizer_calls") or 0) for pair in pairs),
    }


def _print_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    rate = summary["attack_success_rate"]
    rate_s = "n/a" if rate is None else f"{rate:.1%}"
    print(
        f"search none: ASR {summary['attack_success']}/{summary['scored']}={rate_s} "
        f"errors={summary['errors']} victim_calls={summary['victim_calls']}",
        flush=True,
    )


def _pair_key(user_id: str, injection_id: str) -> str:
    return f"{user_id}|{injection_id}"


def _reject_gpt(model: str) -> None:
    if "gpt" in model.casefold():
        raise ValueError("optimizer must not be a GPT model")


if __name__ == "__main__":
    raise SystemExit(main())
