"""Fixed-defense AgentDojo evaluation using native task checkers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from typing import Any, Sequence

from .adapter import load_suite, select_task_ids
from .scoring import aggregate_injection_metrics, injection_succeeded


@dataclass(frozen=True)
class DojoCaseResult:
    user_task_id: str
    injection_task_id: str | None
    utility: bool | None
    security: bool | None
    attack_success: bool | None
    error: str | None = None


@dataclass(frozen=True)
class DojoMetrics:
    n: int
    errors: int
    utility: float | None
    security_rate: float | None
    attack_success_rate: float | None


@dataclass(frozen=True)
class DojoEvaluationReport:
    schema_version: int
    created_at: str
    agentdojo_version: str
    benchmark_version: str
    suite: str
    model: str
    defense: str
    attack: str | None
    selected_user_tasks: tuple[str, ...]
    selected_injection_tasks: tuple[str, ...]
    clean: DojoMetrics
    injected: DojoMetrics
    cases: tuple[DojoCaseResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_fixed_defense(
    pipeline: Any,
    *,
    benchmark_version: str = "v1.2.2",
    suite_name: str = "workspace",
    attack_name: str | None = "injecagent",
    user_task_ids: Sequence[str] | None = None,
    injection_task_ids: Sequence[str] | None = None,
    max_user_tasks: int | None = 3,
    max_injection_tasks: int | None = 3,
    run_clean: bool = True,
    continue_on_error: bool = True,
    model_name: str = "unknown",
    defense_name: str = "none",
) -> DojoEvaluationReport:
    """Run selected tasks in fresh native AgentDojo environments.

    AgentDojo's second return value is the injection checker result:
    ``True`` means the injection goal was achieved. This bridge stores that raw
    value on ``security`` and sets ``attack_success`` to the same boolean.
    Reported ``security_rate`` is the defense rate ``1 - ASR``.
    """
    suite = load_suite(benchmark_version, suite_name)
    selected_users = select_task_ids(
        tuple(suite.user_tasks),
        user_task_ids,
        limit=max_user_tasks,
        kind="user task",
    )
    selected_injections = select_task_ids(
        tuple(suite.injection_tasks),
        injection_task_ids,
        limit=max_injection_tasks,
        kind="injection task",
    )

    attack = None
    if attack_name is not None:
        import contract_heldout_ipi.dojo.chatinject_attack  # noqa: F401
        import contract_heldout_ipi.dojo.pismith_attack  # noqa: F401
        from agentdojo.attacks.attack_registry import ATTACKS, load_attack

        if attack_name not in ATTACKS:
            raise ValueError(
                f"unknown AgentDojo attack {attack_name!r}; choose one of: "
                f"{', '.join(sorted(ATTACKS))}"
            )
        attack = load_attack(attack_name, suite, pipeline)

    cases: list[DojoCaseResult] = []
    if run_clean:
        for user_task_id in selected_users:
            user_task = suite.get_user_task_by_id(user_task_id)
            cases.append(
                _run_case(
                    suite,
                    pipeline,
                    user_task,
                    injection_task=None,
                    injections={},
                    continue_on_error=continue_on_error,
                )
            )

    if attack is not None:
        for user_task_id in selected_users:
            user_task = suite.get_user_task_by_id(user_task_id)
            for injection_task_id in selected_injections:
                injection_task = suite.get_injection_task_by_id(injection_task_id)
                try:
                    injections = attack.attack(user_task, injection_task)
                except Exception as exc:  # attack plugins are an external boundary
                    if not continue_on_error:
                        raise
                    cases.append(
                        DojoCaseResult(
                            user_task_id=user_task_id,
                            injection_task_id=injection_task_id,
                            utility=None,
                            security=None,
                            attack_success=None,
                            error=_safe_error(exc),
                        )
                    )
                    continue
                cases.append(
                    _run_case(
                        suite,
                        pipeline,
                        user_task,
                        injection_task=injection_task,
                        injections=injections,
                        continue_on_error=continue_on_error,
                    )
                )

    clean_cases = [case for case in cases if case.injection_task_id is None]
    injected_cases = [case for case in cases if case.injection_task_id is not None]
    return DojoEvaluationReport(
        schema_version=1,
        created_at=datetime.now(timezone.utc).isoformat(),
        agentdojo_version=version("agentdojo"),
        benchmark_version=benchmark_version,
        suite=suite_name,
        model=model_name,
        defense=defense_name,
        attack=attack_name,
        selected_user_tasks=selected_users,
        selected_injection_tasks=selected_injections if attack is not None else (),
        clean=_metrics(clean_cases, injected=False),
        injected=_metrics(injected_cases, injected=True),
        cases=tuple(cases),
    )


def _run_case(
    suite: Any,
    pipeline: Any,
    user_task: Any,
    *,
    injection_task: Any | None,
    injections: dict[str, str],
    continue_on_error: bool,
) -> DojoCaseResult:
    try:
        utility, security = suite.run_task_with_pipeline(
            pipeline,
            user_task,
            injection_task,
            injections,
        )
    except Exception as exc:  # model providers and Dojo tools are external boundaries
        if not continue_on_error:
            raise
        return DojoCaseResult(
            user_task_id=user_task.ID,
            injection_task_id=(injection_task.ID if injection_task is not None else None),
            utility=None,
            security=None,
            attack_success=None,
            error=_safe_error(exc),
        )
    return DojoCaseResult(
        user_task_id=user_task.ID,
        injection_task_id=(injection_task.ID if injection_task is not None else None),
        utility=bool(utility),
        security=(bool(security) if injection_task is not None else None),
        attack_success=(
            injection_succeeded(security)
            if injection_task is not None
            else None
        ),
    )


def _metrics(cases: Sequence[DojoCaseResult], *, injected: bool) -> DojoMetrics:
    valid = [case for case in cases if case.error is None and case.utility is not None]
    errors = sum(case.error is not None for case in cases)
    if not valid:
        return DojoMetrics(
            n=0,
            errors=errors,
            utility=None,
            security_rate=None,
            attack_success_rate=None,
        )
    utility = sum(bool(case.utility) for case in valid) / len(valid)
    if not injected:
        return DojoMetrics(
            n=len(valid),
            errors=errors,
            utility=utility,
            security_rate=None,
            attack_success_rate=None,
        )
    injection_flags = [
        bool(case.security)
        for case in valid
        if case.security is not None and case.attack_success is not None
    ]
    if not injection_flags:
        security_rate = None
        asr = None
    else:
        asr, security_rate = aggregate_injection_metrics(injection_flags)
    return DojoMetrics(
        n=len(valid),
        errors=errors,
        utility=utility,
        security_rate=security_rate,
        attack_success_rate=asr,
    )


def _safe_error(exc: Exception) -> str:
    text = str(exc).strip()
    cause = exc.__cause__ or exc.__context__
    if cause is not None and str(cause).strip() and str(cause).strip() not in text:
        text = f"{text}; caused by {type(cause).__name__}: {str(cause).strip()}"
    if len(text) > 500:
        text = text[:497] + "..."
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__
