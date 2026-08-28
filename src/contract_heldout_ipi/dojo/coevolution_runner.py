"""Train-time co-evolution of AgentDojo defense harnesses."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from typing import Any, Callable, Sequence

from contract_heldout_ipi.defenses.harness import DefenseHarness

from .dynamic_runner import (
    DojoDynamicReport,
    DojoMetrics,
    DojoNativeResult,
    DojoQualificationRunner,
    _native_metrics,
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


@dataclass(frozen=True)
class DojoCoevolutionReport:
    schema_version: int
    created_at: str
    optimize_enabled: bool
    initial_harness_version: int
    final_harness_version: int
    final_system_prompt: str
    harness_updates: tuple[DojoHarnessUpdateRecord, ...]
    dynamic: DojoDynamicReport

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dynamic"] = self.dynamic.to_dict()
        return payload


PipelineFactory = Callable[[str | None], tuple[Any, str]]


class DojoCoevolutionRunner:
    """Run dynamic qualification while optionally optimizing the defense harness."""

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
        injection_vector_resolver: Callable[[Any], Sequence[str]] | None = None,
    ):
        self.suite = suite
        self.generator = generator
        self.qualification_pipeline = qualification_pipeline
        self.pipeline_factory = pipeline_factory
        self.initial_harness = initial_harness
        self.optimizer = optimizer
        self.qualification_candidates = qualification_candidates
        self.injection_vector_resolver = injection_vector_resolver

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
        runner = DojoQualificationRunner(
            suite=self.suite,
            generator=self.generator,
            qualification_pipeline=self.qualification_pipeline,
            defense_pipeline=defense_pipeline,
            defense_harness=defense_harness,
            qualification_candidates=self.qualification_candidates,
            injection_vector_resolver=self.injection_vector_resolver,
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
        pair_results = []
        for user_id, injection_id in pairs:
            pair_result = runner._run_pair(user_id, injection_id)
            pair_results.append(pair_result)
            defense_result = pair_result.defense_result
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
                        )
                    )
                    harness = update.harness
                    defense_pipeline, defense_harness = self.pipeline_factory(
                        harness.prompt
                    )
                    runner.defense_pipeline = defense_pipeline
                    runner.defense_harness = defense_harness

        pair_results = tuple(pair_results)
        qualified = sum(result.qualified for result in pair_results)
        formal_results = [
            result.defense_result
            for result in pair_results
            if result.defense_result is not None
        ]
        dynamic = DojoDynamicReport(
            schema_version=1,
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
            qualification_rate=(qualified / len(pair_results) if pair_results else None),
            qualification_pairs=len(pair_results),
            qualified_pairs=qualified,
            clean=_native_metrics(
                [result for _, result in clean_results], injected=False
            ),
            admitted=_native_metrics(formal_results, injected=True),
            clean_results=tuple(clean_results),
            pairs=pair_results,
        )
        return DojoCoevolutionReport(
            schema_version=1,
            created_at=dynamic.created_at,
            optimize_enabled=optimize and self.optimizer is not None,
            initial_harness_version=self.initial_harness.version,
            final_harness_version=harness.version,
            final_system_prompt=harness.prompt,
            harness_updates=tuple(updates),
            dynamic=dynamic,
        )


def _execute_native_clean(
    suite: Any, pipeline: Any, user_task: Any
) -> DojoNativeResult:
    from .dynamic_runner import _execute_native

    return _execute_native(
        suite,
        pipeline,
        user_task,
        injection_task=None,
        injections={},
    )
