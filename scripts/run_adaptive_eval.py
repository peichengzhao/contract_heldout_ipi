#!/usr/bin/env python3
"""Run harness-aware attacks and train-time defense adaptation."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Literal, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from contract_heldout_ipi.agent import (  # noqa: E402
    ExperimentModelConfig,
    LLMAgent,
    ModelClientError,
    ModelConfigurationError,
    OpenAICompatibleChatClient,
)
from contract_heldout_ipi.attack import HarnessAwareAttackGenerator  # noqa: E402
from contract_heldout_ipi.contract.loader import load_episodes_dir  # noqa: E402
from contract_heldout_ipi.defenses import (  # noqa: E402
    DefenseHarness,
    HarnessOptimizer,
    get_defense,
)
from contract_heldout_ipi.eval import EvaluationExecutionError, LLMEpisodeJudge  # noqa: E402
from contract_heldout_ipi.eval.adaptive import AdaptiveEvaluationRunner  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate harness-aware attacks, optimize only on train traces, "
            "then freeze the harness for held-out evaluation."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--initial-defense",
        choices=("no-defense", "system-prompt-warning"),
        default="system-prompt-warning",
    )
    parser.add_argument("--train-rounds", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--attack-temperature", type=float, default=0.7)
    parser.add_argument("--defense-temperature", type=float, default=0.0)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = ExperimentModelConfig.from_file(args.config)
    except (ModelConfigurationError, OSError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    initial_defense = get_defense(args.initial_defense)
    initial_harness = DefenseHarness(
        name=initial_defense.name,
        prompt=initial_defense.system_prompt(),
    )
    attack_client = _client(config, "attack", args.attack_temperature, args)
    defense_client = _client(config, "defense", args.defense_temperature, args)
    judge_client = _client(config, "judge", args.judge_temperature, args)
    adaptive_runner = AdaptiveEvaluationRunner(
        attack_generator=HarnessAwareAttackGenerator(attack_client),
        defense_agent_factory=lambda harness: LLMAgent(defense_client, harness),
        score_episode=LLMEpisodeJudge(judge_client).score,
        optimizer=HarnessOptimizer(judge_client),
    )
    episodes = [
        *load_episodes_dir(ROOT / "episodes" / "train"),
        *load_episodes_dir(ROOT / "episodes" / "heldout"),
    ]
    try:
        report = adaptive_runner.run(
            episodes,
            initial_harness=initial_harness,
            train_rounds=args.train_rounds,
        )
    except (EvaluationExecutionError, ModelClientError, ValueError) as exc:
        print(f"Adaptive evaluation error: {exc}", file=sys.stderr)
        return 1

    _write_output(args.output, config, report)
    print(f"Train rounds: {args.train_rounds}")
    print(f"Final harness version: {report.final_harness.version}")
    print(f"Train ASR: {report.train.attack_success_rate:.3f}")
    print(f"Train Utility: {report.train.benign_utility:.3f}")
    print(f"Held-out ASR: {report.heldout.attack_success_rate:.3f}")
    print(f"Held-out Utility: {report.heldout.benign_utility:.3f}")
    print(f"Security Gap: {report.gap.security_gap:.3f}")
    return 0


def _client(
    config: ExperimentModelConfig,
    role: Literal["attack", "defense", "judge"],
    temperature: float,
    args: argparse.Namespace,
) -> OpenAICompatibleChatClient:
    return OpenAICompatibleChatClient(
        config.for_role(role),
        timeout=args.timeout,
        temperature=temperature,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
    )


def _write_output(path: Path, config: ExperimentModelConfig, report: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "models": {
            "attack_model": config.attack_model,
            "defense_model": config.defense_model,
            "judge_model": config.judge_model,
        },
        "report": asdict(report),
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
