#!/usr/bin/env python3
"""Run negotiated attack-defense co-evolution with historical replays."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from contract_heldout_ipi.agent import (  # noqa: E402
    ChatModelClient,
    ExperimentModelConfig,
    LLMAgent,
    ModelClientError,
    ModelConfigurationError,
    ModelRole,
    build_role_client,
)
from contract_heldout_ipi.attack import (  # noqa: E402
    AttackCritic,
    HarnessAwareAttackGenerator,
)
from contract_heldout_ipi.contract.loader import load_episodes_dir  # noqa: E402
from contract_heldout_ipi.defenses import (  # noqa: E402
    DefenseHarness,
    HarnessOptimizer,
    get_defense,
)
from contract_heldout_ipi.eval import EvaluationExecutionError, LLMEpisodeJudge  # noqa: E402
from contract_heldout_ipi.game import (  # noqa: E402
    ATTACKER_ADVANTAGE_CONSTITUTION,
    BALANCED_CONSTITUTION,
    GameNegotiator,
    NegotiatedGameRunner,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Negotiate structured rules, generate white-box attacks, evolve the "
            "defense, replay historical attacks, then freeze for held-out eval."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--initial-defense",
        choices=("no-defense", "system-prompt-warning"),
        default="no-defense",
    )
    parser.add_argument(
        "--rule-profile",
        choices=("attacker-advantage", "balanced"),
        default="attacker-advantage",
    )
    parser.add_argument("--train-rounds", type=int, default=3)
    parser.add_argument("--negotiation-rounds", type=int, default=4)
    parser.add_argument("--rule-judge-attempts", type=int, default=3)
    parser.add_argument("--optimizer-attempts", type=int, default=3)
    parser.add_argument("--attack-attempts", type=int, default=3)
    parser.add_argument("--critic-attempts", type=int, default=2)
    parser.add_argument("--attack-candidates", type=int, default=5)
    parser.add_argument("--max-archive-size", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--deepseek-max-tokens", type=int, default=8192)
    parser.add_argument("--attack-temperature", type=float, default=0.7)
    parser.add_argument("--defense-temperature", type=float, default=0.0)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--rule-judge-temperature", type=float, default=0.0)
    parser.add_argument("--optimizer-temperature", type=float, default=0.0)
    parser.add_argument("--task-agent-temperature", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = ExperimentModelConfig.from_file(args.config)
    except (ModelConfigurationError, OSError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    attack_client = _client(config, "attack", args.attack_temperature, args)
    defense_client = _client(config, "defense", args.defense_temperature, args)
    judge_client = _client(config, "judge", args.judge_temperature, args)
    rule_judge_client = _client(
        config, "rule_judge", args.rule_judge_temperature, args
    )
    optimizer_client = _client(
        config, "optimizer", args.optimizer_temperature, args
    )
    task_agent_client = _client(
        config, "task_agent", args.task_agent_temperature, args
    )
    initial = get_defense(args.initial_defense)
    constitution = (
        ATTACKER_ADVANTAGE_CONSTITUTION
        if args.rule_profile == "attacker-advantage"
        else BALANCED_CONSTITUTION
    )
    game = NegotiatedGameRunner(
        negotiator=GameNegotiator(
            attack_client,
            defense_client,
            rule_judge_client,
            constitution=constitution,
            max_judge_attempts=args.rule_judge_attempts,
        ),
        attack_generator=HarnessAwareAttackGenerator(
            attack_client, max_attempts=args.attack_attempts
        ),
        defense_agent_factory=lambda harness: LLMAgent(task_agent_client, harness),
        score_episode=LLMEpisodeJudge(judge_client).score,
        optimizer=HarnessOptimizer(
            optimizer_client, max_attempts=args.optimizer_attempts
        ),
        attack_critic=AttackCritic(
            attack_client, max_attempts=args.critic_attempts
        ),
        max_archive_size=args.max_archive_size,
        qualification_candidates=args.attack_candidates,
    )
    episodes = [
        *load_episodes_dir(ROOT / "episodes" / "train"),
        *load_episodes_dir(ROOT / "episodes" / "heldout"),
    ]
    try:
        report = game.run(
            episodes,
            initial_harness=DefenseHarness(initial.name, initial.system_prompt()),
            train_rounds=args.train_rounds,
            negotiation_rounds=args.negotiation_rounds,
        )
    except (EvaluationExecutionError, ModelClientError, ValueError) as exc:
        print(f"Negotiated game error: {exc}", file=sys.stderr)
        return 1

    _write_output(args.output, config, args, report)
    print(f"Final harness version: {report.final_harness.version}")
    print(f"Negotiated train matches: {len(report.train_history)}")
    print(f"Attack qualification failures: {len(report.qualification_failures)}")
    duplicate_rejections = sum(
        item.qualification_duplicate_rejections for item in report.train_history
    ) + sum(
        item.qualification_duplicate_rejections for item in report.heldout_results
    ) + sum(
        item.duplicate_rejections for item in report.qualification_failures
    )
    print(f"Duplicate attack candidates rejected: {duplicate_rejections}")
    generation_failures = sum(
        len(item.qualification_generation_failures)
        for item in report.train_history
    ) + sum(
        len(item.qualification_generation_failures)
        for item in report.heldout_results
    ) + sum(
        len(item.generation_failures) for item in report.qualification_failures
    )
    print(f"Attack generation fallbacks: {generation_failures}")
    print(f"Historical league replays: {len(report.league_replays)}")
    print(
        "Final-round train ASR / Utility: "
        f"{report.train.attack_success_rate:.3f} / {report.train.benign_utility:.3f} "
        f"(n={report.train.n})"
    )
    print(
        "All admitted train ASR / Utility: "
        f"{report.train_all.attack_success_rate:.3f} / "
        f"{report.train_all.benign_utility:.3f} (n={report.train_all.n})"
    )
    print(
        "Held-out ASR / Utility: "
        f"{report.heldout.attack_success_rate:.3f} / "
        f"{report.heldout.benign_utility:.3f}"
        f" (n={report.heldout.n})"
    )
    if report.gap.security_gap is None:
        print("Transfer gap: N/A (train or held-out has no admitted matches)")
    else:
        print(
            "Security / Utility gap: "
            f"{report.gap.security_gap:.3f} / {report.gap.utility_gap:.3f}"
        )
    print(
        "Final-harness archive replay ASR / Utility: "
        f"{report.final_harness_replay.attack_success_rate:.3f} / "
        f"{report.final_harness_replay.benign_utility:.3f} "
        f"(n={report.final_harness_replay.n})"
    )
    print(f"Wrote {args.output}")
    return 0


def _client(
    config: ExperimentModelConfig,
    role: ModelRole,
    temperature: float,
    args: argparse.Namespace,
) -> ChatModelClient:
    return build_role_client(
        config,
        role,
        timeout=args.timeout,
        temperature=temperature,
        max_tokens=(
            args.deepseek_max_tokens
            if _role_model(config, role).casefold().startswith("deepseek")
            else args.max_tokens
        ),
        max_retries=args.max_retries,
    )


def _role_model(config: ExperimentModelConfig, role: ModelRole) -> str:
    if role == "attack":
        return config.attack_model
    return config.for_role(role).model


def _write_output(
    path: Path,
    config: ExperimentModelConfig,
    args: argparse.Namespace,
    report: object,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": {
            "name": "negotiated-email-game-v1",
            "rule_profile": args.rule_profile,
            "train_rounds": args.train_rounds,
            "negotiation_rounds": args.negotiation_rounds,
            "rule_judge_attempts": args.rule_judge_attempts,
            "optimizer_attempts": args.optimizer_attempts,
            "attack_attempts": args.attack_attempts,
            "critic_attempts": args.critic_attempts,
            "attack_candidates": args.attack_candidates,
            "max_archive_size": args.max_archive_size,
            "max_tokens": args.max_tokens,
            "deepseek_max_tokens": args.deepseek_max_tokens,
        },
        "models": {
            "attack_model": config.attack_model,
            "attack_provider": config.attack_provider,
            "defense_model": config.defense_model,
            "judge_model": config.judge_model,
            "rule_judge_model": config.for_role("rule_judge").model,
            "optimizer_model": config.for_role("optimizer").model,
            "task_agent_model": config.for_role("task_agent").model,
        },
        "report": _jsonable(report),
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _jsonable(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
