"""Train-time bidirectional co-evolution for AgentDojo attacks and defenses."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from typing import Any, Callable, Sequence

from contract_heldout_ipi.defenses.harness import DefenseHarness

from .dynamic_runner import (
    DojoDynamicReport,
    DojoDynamicPairResult,
    DojoMetrics,
    DojoNativeResult,
    DojoQualificationRunner,
    DojoReplaySummary,
    _admitted_metrics,
    _execute_native,
    _execute_replays,
    _native_metrics,
    _representative_result,
    _success_count,
    _summarize_replays,
)
from .optimizer import DojoHarnessOptimizer, should_update_harness


@dataclass(frozen=True)
class DojoHarnessUpdateRecord:
    user_task_id: str
    injection_task_id: str
    from_version: int
    to_version: int
    rationale: str
    trigger: str
    round_index: int


@dataclass(frozen=True)
class DojoArchivedAttack:
    user_task_id: str
    injection_task_id: str
    strategy: str
    payload: str
    target_vectors: tuple[str, ...]
    admitted_injections: dict[str, str]
    round_index: int


@dataclass(frozen=True)
class DojoLeagueReplayResult:
    user_task_id: str
    injection_task_id: str
    source_round_index: int
    harness_version: int
    defense_result: DojoNativeResult
    defense_replay_summary: DojoReplaySummary


@dataclass(frozen=True)
class DojoRoundSummary:
    round_index: int
    qualification_pairs: int
    qualified_pairs: int
    qualification_rate: float | None
    harness_version: int
    archive_size: int


@dataclass(frozen=True)
class DojoCoevolutionReport:
    schema_version: int
    created_at: str
    optimize_enabled: bool
    train_rounds: int
    max_archive_size: int
    initial_harness_version: int
    final_harness_version: int
    final_system_prompt: str
    harness_updates: tuple[DojoHarnessUpdateRecord, ...]
    archive: tuple[DojoArchivedAttack, ...]
    league_replays: tuple[DojoLeagueReplayResult, ...]
    final_league_replay: DojoMetrics
    round_summaries: tuple[DojoRoundSummary, ...]
    dynamic: DojoDynamicReport

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dynamic"] = self.dynamic.to_dict()
        return payload


PipelineFactory = Callable[[str | None], tuple[Any, str]]


class DojoCoevolutionRunner:
    """Run multi-round dynamic attacks with defense harness and attack-archive co-evolution."""

    def __init__(
        self,
        *,
        suite: Any,
        generator: Any,
        qualification_pipeline: Any,
        pipeline_factory: PipelineFactory,
        initial_harness: DefenseHarness,
        optimizer: DojoHarnessOptimizer | None = None,
        qualification_candidates: int = 3,
        qualification_replay_runs: int = 3,
        qualification_success_min: int | None = None,
        defense_replay_runs: int = 5,
        injection_vector_resolver: Callable[[Any], Sequence[str]] | None = None,
        train_rounds: int = 1,
        max_archive_size: int = 20,
        attack_memory_limit: int = 10,
    ):
        if train_rounds < 1:
            raise ValueError("train_rounds must be at least 1")
        if max_archive_size < 1:
            raise ValueError("max_archive_size must be at least 1")
        if attack_memory_limit < 1:
            raise ValueError("attack_memory_limit must be at least 1")
        self.suite = suite
        self.generator = generator
        self.qualification_pipeline = qualification_pipeline
        self.pipeline_factory = pipeline_factory
        self.initial_harness = initial_harness
        self.optimizer = optimizer
        self.qualification_candidates = qualification_candidates
        self.qualification_replay_runs = qualification_replay_runs
        self.qualification_success_min = qualification_success_min
        self.defense_replay_runs = defense_replay_runs
        self.injection_vector_resolver = injection_vector_resolver
        self.train_rounds = train_rounds
        self.max_archive_size = max_archive_size
        self.attack_memory_limit = attack_memory_limit

    def run(
        self,
        *,
        user_task_ids: Sequence[str],
        injection_task_ids: Sequence[str],
        benchmark_version: str,
        suite_name: str,
        split: str,
        attack_model: str,
        task_model: str,
        defense_name: str,
        max_pairs: int | None = None,
        run_clean: bool = True,
        optimize: bool = True,
    ) -> DojoCoevolutionReport:
        if max_pairs is not None and max_pairs < 1:
            raise ValueError("max_pairs must be positive")
        selected_users = tuple(user_task_ids)
        selected_injections = tuple(injection_task_ids)
        pairs = [
            (user_id, injection_id)
            for user_id in selected_users
            for injection_id in selected_injections
        ]
        if max_pairs is not None:
            pairs = pairs[:max_pairs]
        active_users = tuple(dict.fromkeys(user_id for user_id, _ in pairs))
        active_injections = tuple(
            dict.fromkeys(injection_id for _, injection_id in pairs)
        )

        harness = self.initial_harness
        defense_pipeline, defense_harness = self.pipeline_factory(harness.prompt)
        attack_memory: list[dict[str, Any]] = []
        runner = DojoQualificationRunner(
            suite=self.suite,
            generator=self.generator,
            qualification_pipeline=self.qualification_pipeline,
            defense_pipeline=defense_pipeline,
            defense_harness=defense_harness,
            qualification_candidates=self.qualification_candidates,
            qualification_replay_runs=self.qualification_replay_runs,
            qualification_success_min=self.qualification_success_min,
            defense_replay_runs=self.defense_replay_runs,
            injection_vector_resolver=self.injection_vector_resolver,
            attack_memory=attack_memory,
        )

        clean_results: list[tuple[str, DojoNativeResult]] = []
        if run_clean:
            for user_id in active_users:
                user_task = self.suite.get_user_task_by_id(user_id)
                clean_results.append(
                    (
                        user_id,
                        _execute_native_clean(
                            self.suite, runner.defense_pipeline, user_task
                        ),
                    )
                )

        updates: list[DojoHarnessUpdateRecord] = []
        archive: list[DojoArchivedAttack] = []
        league_replays: list[DojoLeagueReplayResult] = []
        round_summaries: list[DojoRoundSummary] = []
        last_pair_results: tuple[DojoDynamicPairResult, ...] = ()

        for round_index in range(self.train_rounds):
            round_pairs: list[DojoDynamicPairResult] = []
            for user_id, injection_id in pairs:
                pair_result = runner._run_pair(user_id, injection_id)
                round_pairs.append(pair_result)
                _extend_attack_memory(
                    attack_memory,
                    pair_result,
                    round_index=round_index,
                    limit=self.attack_memory_limit,
                )
                defense_result = pair_result.defense_result
                if pair_result.admitted_plan is not None and pair_result.admitted_injections is not None:
                    archive.append(
                        DojoArchivedAttack(
                            user_task_id=user_id,
                            injection_task_id=injection_id,
                            strategy=pair_result.admitted_plan.strategy,
                            payload=pair_result.admitted_plan.payload,
                            target_vectors=pair_result.admitted_plan.target_vectors,
                            admitted_injections=dict(pair_result.admitted_injections),
                            round_index=round_index,
                        )
                    )
                    archive[:] = archive[-self.max_archive_size :]
                if (
                    optimize
                    and self.optimizer is not None
                    and pair_result.admitted_plan is not None
                    and should_update_harness(defense_result)
                    and defense_result is not None
                ):
                    user_task = self.suite.get_user_task_by_id(user_id)
                    injection_task = self.suite.get_injection_task_by_id(injection_id)
                    update = self.optimizer.optimize(
                        harness=harness,
                        user_task_prompt=user_task.PROMPT,
                        injection_goal=injection_task.GOAL,
                        attack_plan=pair_result.admitted_plan,
                        result=defense_result,
                        user_task_id=user_id,
                        injection_task_id=injection_id,
                    )
                    if update.harness.version != harness.version:
                        updates.append(
                            DojoHarnessUpdateRecord(
                                user_task_id=user_id,
                                injection_task_id=injection_id,
                                from_version=harness.version,
                                to_version=update.harness.version,
                                rationale=update.rationale,
                                trigger=update.trigger,
                                round_index=round_index,
                            )
                        )
                        harness = update.harness
                        defense_pipeline, defense_harness = self.pipeline_factory(
                            harness.prompt
                        )
                        runner.defense_pipeline = defense_pipeline
                        runner.defense_harness = defense_harness
                        league_replays.extend(
                            _replay_archive(
                                self.suite,
                                archive,
                                defense_pipeline,
                                defense_replay_runs=self.defense_replay_runs,
                                harness_version=harness.version,
                            )
                        )

            qualified = sum(result.qualified for result in round_pairs)
            round_summaries.append(
                DojoRoundSummary(
                    round_index=round_index,
                    qualification_pairs=len(round_pairs),
                    qualified_pairs=qualified,
                    qualification_rate=(
                        qualified / len(round_pairs) if round_pairs else None
                    ),
                    harness_version=harness.version,
                    archive_size=len(archive),
                )
            )
            last_pair_results = tuple(round_pairs)

        if archive:
            final_batch = _replay_archive(
                self.suite,
                archive,
                runner.defense_pipeline,
                defense_replay_runs=self.defense_replay_runs,
                harness_version=harness.version,
            )
            league_replays.extend(final_batch)
            final_league_replay = _league_metrics(final_batch)
        else:
            final_league_replay = DojoMetrics(0, 0, None, None, None)

        qualified = sum(result.qualified for result in last_pair_results)
        dynamic = DojoDynamicReport(
            schema_version=2,
            created_at=datetime.now(timezone.utc).isoformat(),
            agentdojo_version=version("agentdojo"),
            benchmark_version=benchmark_version,
            suite=suite_name,
            split=split,
            attack_model=attack_model,
            task_model=task_model,
            defense=defense_name,
            defense_harness=runner.defense_harness,
            selected_user_tasks=active_users,
            selected_injection_tasks=active_injections,
            qualification_candidates=self.qualification_candidates,
            qualification_replay_runs=self.qualification_replay_runs,
            qualification_success_min=runner.qualification_success_min,
            defense_replay_runs=self.defense_replay_runs,
            qualification_rate=(qualified / len(last_pair_results) if last_pair_results else None),
            qualification_pairs=len(last_pair_results),
            qualified_pairs=qualified,
            clean=_native_metrics(
                [result for _, result in clean_results], injected=False
            ),
            admitted=_admitted_metrics(last_pair_results),
            clean_results=tuple(clean_results),
            pairs=last_pair_results,
        )
        return DojoCoevolutionReport(
            schema_version=2,
            created_at=dynamic.created_at,
            optimize_enabled=optimize and self.optimizer is not None,
            train_rounds=self.train_rounds,
            max_archive_size=self.max_archive_size,
            initial_harness_version=self.initial_harness.version,
            final_harness_version=harness.version,
            final_system_prompt=harness.prompt,
            harness_updates=tuple(updates),
            archive=tuple(archive),
            league_replays=tuple(league_replays),
            final_league_replay=final_league_replay,
            round_summaries=tuple(round_summaries),
            dynamic=dynamic,
        )


def _extend_attack_memory(
    attack_memory: list[dict[str, Any]],
    pair: DojoDynamicPairResult,
    *,
    round_index: int,
    limit: int,
) -> None:
    for attempt in pair.qualification_attempts:
        if attempt.plan is None or attempt.result is None:
            continue
        attack_memory.append(
            {
                "round_index": round_index,
                "user_task_id": pair.user_task_id,
                "injection_task_id": pair.injection_task_id,
                "strategy": attempt.plan.strategy,
                "payload_preview": attempt.plan.payload[:240],
                "attack_success": attempt.result.attack_success,
                "qualification_successes": _success_count(
                    attempt.qualification_replays
                ),
                "observed_trajectory": attempt.result.trace.compact(),
            }
        )
    if pair.admitted_plan is not None and pair.defense_result is not None:
        summary = pair.defense_replay_summary
        attack_memory.append(
            {
                "round_index": round_index,
                "user_task_id": pair.user_task_id,
                "injection_task_id": pair.injection_task_id,
                "strategy": pair.admitted_plan.strategy,
                "payload_preview": pair.admitted_plan.payload[:240],
                "admitted": True,
                "defense_attack_success": pair.defense_result.attack_success,
                "defense_replay_asr": (
                    summary.attack_success_rate if summary is not None else None
                ),
            }
        )
    if len(attack_memory) > limit:
        del attack_memory[:-limit]


def _replay_archive(
    suite: Any,
    archive: Sequence[DojoArchivedAttack],
    defense_pipeline: Any,
    *,
    defense_replay_runs: int,
    harness_version: int,
) -> list[DojoLeagueReplayResult]:
    results: list[DojoLeagueReplayResult] = []
    for item in archive:
        user_task = suite.get_user_task_by_id(item.user_task_id)
        injection_task = suite.get_injection_task_by_id(item.injection_task_id)
        replays = _execute_replays(
            suite,
            defense_pipeline,
            user_task,
            injection_task=injection_task,
            injections=dict(item.admitted_injections),
            runs=defense_replay_runs,
        )
        summary = _summarize_replays(replays)
        defense_result = _representative_result(replays, summary)
        results.append(
            DojoLeagueReplayResult(
                user_task_id=item.user_task_id,
                injection_task_id=item.injection_task_id,
                source_round_index=item.round_index,
                harness_version=harness_version,
                defense_result=defense_result,
                defense_replay_summary=summary,
            )
        )
    return results


def _league_metrics(replays: Sequence[DojoLeagueReplayResult]) -> DojoMetrics:
    summaries = [item.defense_replay_summary for item in replays]
    if not summaries:
        return DojoMetrics(0, 0, None, None, None)
    errors = sum(summary.errors for summary in summaries)
    utility_rates = [
        summary.utility_rate
        for summary in summaries
        if summary.utility_rate is not None
    ]
    attack_rates = [
        summary.attack_success_rate
        for summary in summaries
        if summary.attack_success_rate is not None
    ]
    utility = sum(utility_rates) / len(utility_rates) if utility_rates else None
    if not attack_rates:
        return DojoMetrics(len(summaries), errors, utility, None, None)
    from .scoring import aggregate_injection_rates

    asr, defense_rate = aggregate_injection_rates(attack_rates)
    return DojoMetrics(len(summaries), errors, utility, defense_rate, asr)


def _execute_native_clean(
    suite: Any, pipeline: Any, user_task: Any
) -> DojoNativeResult:
    return _execute_native(
        suite,
        pipeline,
        user_task,
        injection_task=None,
        injections={},
    )
