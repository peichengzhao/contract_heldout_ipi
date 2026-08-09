"""Batch execution and train-to-held-out metric aggregation."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias

from contract_heldout_ipi.contract.models import EpisodeContract

from .judge import EpisodeScore, judge_episode
from .metrics import SplitMetrics, TransferGap
from .trace import EpisodeRun

if TYPE_CHECKING:
    from contract_heldout_ipi.agent.protocol import AgentPolicy
    from contract_heldout_ipi.agent.runner import EpisodeRunner

AgentFactory: TypeAlias = Callable[[EpisodeContract], "AgentPolicy"]
ScoreFunction: TypeAlias = Callable[[EpisodeContract, EpisodeRun], EpisodeScore]


class EvaluationExecutionError(RuntimeError):
    """Raised when infrastructure/model failure would invalidate metrics."""


@dataclass(frozen=True)
class EpisodeEvaluation:
    """Execution trace and score for one episode."""

    episode_id: str
    split: Literal["train", "heldout"]
    run: EpisodeRun
    score: EpisodeScore
    untrusted_content: object


@dataclass(frozen=True)
class EvaluationReport:
    """Per-episode results and aggregate held-out transfer metrics."""

    episodes: tuple[EpisodeEvaluation, ...]
    train: SplitMetrics
    heldout: SplitMetrics
    gap: TransferGap


def evaluate_episodes(
    episodes: Iterable[EpisodeContract],
    agent_factory: AgentFactory,
    *,
    runner: EpisodeRunner | None = None,
    score_episode: ScoreFunction = judge_episode,
) -> EvaluationReport:
    """Run and score a contract set containing train and held-out episodes."""
    from contract_heldout_ipi.agent.runner import EpisodeRunner

    episode_list = list(episodes)
    _validate_episode_set(episode_list)
    episode_runner = runner or EpisodeRunner()

    results: list[EpisodeEvaluation] = []
    for episode in episode_list:
        run = episode_runner.run(episode, agent_factory(episode))
        if run.terminated_reason == "agent_error":
            detail = f": {run.agent_error}" if run.agent_error else ""
            raise EvaluationExecutionError(
                f"agent failed in episode '{episode.episode_id}'{detail}"
            )
        try:
            score = score_episode(episode, run)
        except Exception as exc:
            raise EvaluationExecutionError(
                f"judge failed in episode '{episode.episode_id}': "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        results.append(
            EpisodeEvaluation(
                episode_id=episode.episode_id,
                split=episode.split,
                run=run,
                score=score,
                untrusted_content=episode.untrusted_context.content,
            )
        )

    train = SplitMetrics.from_scores(
        result.score for result in results if result.split == "train"
    )
    heldout = SplitMetrics.from_scores(
        result.score for result in results if result.split == "heldout"
    )
    return EvaluationReport(
        episodes=tuple(results),
        train=train,
        heldout=heldout,
        gap=TransferGap.from_splits(train, heldout),
    )


def _validate_episode_set(episodes: list[EpisodeContract]) -> None:
    if not episodes:
        raise ValueError("evaluation requires at least one episode")

    episode_ids = [episode.episode_id for episode in episodes]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("evaluation episode_id values must be unique")

    splits = {episode.split for episode in episodes}
    missing = {"train", "heldout"} - splits
    if missing:
        missing_names = ", ".join(sorted(missing))
        raise ValueError(f"evaluation requires split(s): {missing_names}")
