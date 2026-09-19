"""Small, version-pinned bridge to AgentDojo task suites."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Sequence


class AgentDojoDependencyError(RuntimeError):
    """Raised when the optional AgentDojo dependency is unavailable."""


@dataclass(frozen=True)
class DojoTask:
    task_id: str
    prompt: str


@dataclass(frozen=True)
class DojoManifest:
    benchmark_version: str
    suite: str
    agentdojo_version: str
    user_tasks: tuple[DojoTask, ...]
    injection_tasks: tuple[DojoTask, ...]
    tool_names: tuple[str, ...]
    injection_vectors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def require_agentdojo() -> None:
    try:
        version("agentdojo")
    except PackageNotFoundError as exc:
        raise AgentDojoDependencyError(
            "AgentDojo is not installed; run `pip install -e '.[dojo]'`"
        ) from exc


def load_suite(benchmark_version: str = "v1.2.2", suite_name: str = "workspace") -> Any:
    require_agentdojo()
    from agentdojo.task_suite.load_suites import get_suite, get_suites

    suites = get_suites(benchmark_version)
    if not suites:
        raise ValueError(f"unknown AgentDojo benchmark version: {benchmark_version}")
    if suite_name not in suites:
        choices = ", ".join(sorted(suites))
        raise ValueError(
            f"unknown AgentDojo suite {suite_name!r} for {benchmark_version}; "
            f"choose one of: {choices}"
        )
    return get_suite(benchmark_version, suite_name)


def build_manifest(
    benchmark_version: str = "v1.2.2", suite_name: str = "workspace"
) -> DojoManifest:
    suite = load_suite(benchmark_version, suite_name)
    return DojoManifest(
        benchmark_version=benchmark_version,
        suite=suite_name,
        agentdojo_version=version("agentdojo"),
        user_tasks=tuple(
            DojoTask(task_id=task_id, prompt=task.PROMPT)
            for task_id, task in suite.user_tasks.items()
        ),
        injection_tasks=tuple(
            DojoTask(task_id=task_id, prompt=task.GOAL)
            for task_id, task in suite.injection_tasks.items()
        ),
        tool_names=tuple(tool.name for tool in suite.tools),
        injection_vectors=tuple(suite.get_injection_vector_defaults()),
    )


def expand_injectable_vectors(
    suite: Any,
    primary_vectors: Sequence[str],
    *,
    include_suite_vectors: bool = True,
) -> tuple[str, ...]:
    """Union task-specific vectors with the full suite injection surface.

    ``primary_vectors`` are the vectors AgentDojo marks as reachable during a
    correct no-attack execution. G1 expands the attacker's writable surface to
    every workspace injection slot so multi-vector plans are not rejected upfront.
    """
    primary = tuple(dict.fromkeys(primary_vectors))
    if not include_suite_vectors:
        return primary
    defaults = suite.get_injection_vector_defaults()
    if isinstance(defaults, dict):
        suite_vectors = tuple(defaults.keys())
    else:
        suite_vectors = tuple(defaults)
    return tuple(dict.fromkeys([*primary, *suite_vectors]))


def select_task_ids(
    available: Sequence[str],
    requested: Sequence[str] | None,
    *,
    limit: int | None,
    kind: str,
) -> tuple[str, ...]:
    if limit is not None and limit < 1:
        raise ValueError(f"{kind} limit must be positive")
    available_tuple = tuple(available)
    if requested:
        missing = [task_id for task_id in requested if task_id not in available_tuple]
        if missing:
            raise ValueError(f"unknown {kind} IDs: {', '.join(missing)}")
        return tuple(dict.fromkeys(requested))
    if limit is None:
        return available_tuple
    return available_tuple[:limit]
