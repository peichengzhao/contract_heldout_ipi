"""Train-time adversarial adaptation with a frozen held-out evaluation."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from contract_heldout_ipi.agent.protocol import AgentPolicy
from contract_heldout_ipi.agent.runner import EpisodeRunner
from contract_heldout_ipi.attack.adaptive import (
    AdaptiveAttackGeneration,
    HarnessAwareAttackGenerator,
)
from contract_heldout_ipi.contract.models import EpisodeContract
from contract_heldout_ipi.defenses.harness import DefenseHarness
from contract_heldout_ipi.defenses.optimizer import HarnessOptimizer

from .evaluator import EvaluationExecutionError
from .judge import EpisodeScore
from .metrics import SplitMetrics, TransferGap
from .trace import EpisodeRun

DefenseAgentFactory = Callable[[DefenseHarness], AgentPolicy]
ScoreFunction = Callable[[EpisodeContract, EpisodeRun], EpisodeScore]


@dataclass(frozen=True)
class AdaptiveEpisodeResult:
    """One attack/defense interaction and optional train-time prompt update."""

    episode_id: str
    split: str
    round_index: int
    harness_before: DefenseHarness
    harness_after: DefenseHarness | None
    optimizer_rationale: str | None
    attack_goal: str
    attack_payload: str
    attack_success_condition: dict[str, Any]
    run: EpisodeRun
    score: EpisodeScore


@dataclass(frozen=True)
class AdaptiveEvaluationReport:
    """Adaptation history plus final train-to-held-out metrics."""

    train_history: tuple[AdaptiveEpisodeResult, ...]
    heldout_results: tuple[AdaptiveEpisodeResult, ...]
    final_harness: DefenseHarness
    train: SplitMetrics
    heldout: SplitMetrics
    gap: TransferGap


class AdaptiveEvaluationRunner:
    """Optimize only on train traces, then freeze the harness for held-out."""

    def __init__(
        self,
        *,
        attack_generator: HarnessAwareAttackGenerator,
        defense_agent_factory: DefenseAgentFactory,
        score_episode: ScoreFunction,
        optimizer: HarnessOptimizer,
        runner: EpisodeRunner | None = None,
    ):
        self.attack_generator = attack_generator
        self.defense_agent_factory = defense_agent_factory
        self.score_episode = score_episode
        self.optimizer = optimizer
        self.runner = runner or EpisodeRunner()

    def run(
        self,
        episodes: Iterable[EpisodeContract],
        *,
        initial_harness: DefenseHarness,
        train_rounds: int = 1,
    ) -> AdaptiveEvaluationReport:
        if train_rounds < 1:
            raise ValueError("train_rounds must be at least 1")
        train_episodes, heldout_episodes = _split_episodes(episodes)
        harness = initial_harness
        train_history: list[AdaptiveEpisodeResult] = []

        for round_index in range(train_rounds):
            for episode in train_episodes:
                generation = self.attack_generator.generate(
                    episode, defense_harness=harness.prompt
                )
                run, score = self._execute(generation, harness)
                update = self.optimizer.optimize(
                    generation.episode,
                    harness,
                    generation.plan,
                    run,
                    score,
                )
                train_history.append(
                    _result(
                        generation,
                        run,
                        score,
                        round_index=round_index,
                        harness_before=harness,
                        harness_after=update.harness,
                        optimizer_rationale=update.rationale,
                    )
                )
                harness = update.harness

        heldout_results: list[AdaptiveEpisodeResult] = []
        for episode in heldout_episodes:
            generation = self.attack_generator.generate(
                episode, defense_harness=harness.prompt
            )
            run, score = self._execute(generation, harness)
            heldout_results.append(
                _result(
                    generation,
                    run,
                    score,
                    round_index=train_rounds,
                    harness_before=harness,
                    harness_after=None,
                    optimizer_rationale=None,
                )
            )

        final_train_scores = [
            result.score
            for result in train_history
            if result.round_index == train_rounds - 1
        ]
        train_metrics = SplitMetrics.from_scores(final_train_scores)
        heldout_metrics = SplitMetrics.from_scores(
            result.score for result in heldout_results
        )
        return AdaptiveEvaluationReport(
            train_history=tuple(train_history),
            heldout_results=tuple(heldout_results),
            final_harness=harness,
            train=train_metrics,
            heldout=heldout_metrics,
            gap=TransferGap.from_splits(train_metrics, heldout_metrics),
        )

    def _execute(
        self,
        generation: AdaptiveAttackGeneration,
        harness: DefenseHarness,
    ) -> tuple[EpisodeRun, EpisodeScore]:
        run = self.runner.run(
            generation.episode, self.defense_agent_factory(harness)
        )
        if run.terminated_reason == "agent_error":
            detail = f": {run.agent_error}" if run.agent_error else ""
            raise EvaluationExecutionError(
                f"defense agent failed in episode '{generation.episode.episode_id}'"
                f"{detail}"
            )
        try:
            score = self.score_episode(generation.episode, run)
        except Exception as exc:
            raise EvaluationExecutionError(
                f"judge failed in episode '{generation.episode.episode_id}': "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return run, score


def _split_episodes(
    episodes: Iterable[EpisodeContract],
) -> tuple[list[EpisodeContract], list[EpisodeContract]]:
    episode_list = list(episodes)
    train = [episode for episode in episode_list if episode.split == "train"]
    heldout = [episode for episode in episode_list if episode.split == "heldout"]
    if not train or not heldout:
        raise ValueError("adaptive evaluation requires train and heldout episodes")
    return train, heldout


def _result(
    generation: AdaptiveAttackGeneration,
    run: EpisodeRun,
    score: EpisodeScore,
    *,
    round_index: int,
    harness_before: DefenseHarness,
    harness_after: DefenseHarness | None,
    optimizer_rationale: str | None,
) -> AdaptiveEpisodeResult:
    return AdaptiveEpisodeResult(
        episode_id=generation.episode.episode_id,
        split=generation.episode.split,
        round_index=round_index,
        harness_before=harness_before,
        harness_after=harness_after,
        optimizer_rationale=optimizer_rationale,
        attack_goal=generation.plan.attack_goal,
        attack_payload=generation.plan.payload,
        attack_success_condition=generation.plan.success_condition.model_dump(
            mode="json"
        ),
        run=run,
        score=score,
    )
