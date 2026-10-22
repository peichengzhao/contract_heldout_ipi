from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("agentdojo")

from contract_heldout_ipi.agent import ExperimentModelConfig
from contract_heldout_ipi.dojo import (
    build_dojo_pipeline,
    build_manifest,
    select_task_ids,
)
from contract_heldout_ipi.dojo.evaluator import (
    DojoCaseResult,
    _metrics,
    _run_case,
    _safe_error,
)


def test_workspace_manifest_uses_real_agentdojo_suite() -> None:
    manifest = build_manifest("v1.2.2", "workspace")

    assert manifest.agentdojo_version == "0.1.35"
    assert len(manifest.user_tasks) == 40
    assert len(manifest.injection_tasks) == 14
    assert len(manifest.tool_names) == 24
    assert manifest.user_tasks[0].task_id == "user_task_0"
    assert manifest.injection_tasks[0].task_id == "injection_task_0"


def test_select_task_ids_validates_and_limits() -> None:
    available = ("task_0", "task_1", "task_2")

    assert select_task_ids(available, None, limit=2, kind="task") == (
        "task_0",
        "task_1",
    )
    assert select_task_ids(
        available, ("task_2", "task_2"), limit=1, kind="task"
    ) == ("task_2",)
    with pytest.raises(ValueError, match="unknown task IDs"):
        select_task_ids(available, ("missing",), limit=1, kind="task")


def test_injected_metrics_use_native_injection_checker_and_exclude_errors() -> None:
    metrics = _metrics(
        [
            DojoCaseResult("u0", "i0", True, True, True),
            DojoCaseResult("u1", "i0", True, False, False),
            DojoCaseResult("u2", "i0", None, None, None, "timeout"),
        ],
        injected=True,
    )

    assert metrics.n == 2
    assert metrics.errors == 1
    assert metrics.utility == 1.0
    assert metrics.attack_success_rate == 0.5
    assert metrics.security_rate == 0.5


def test_run_case_delegates_to_native_suite_checker() -> None:
    class Task:
        ID = "user_task_7"

    class Injection:
        ID = "injection_task_2"

    class Suite:
        def __init__(self) -> None:
            self.called_with = None

        def run_task_with_pipeline(self, *args):
            self.called_with = args
            return True, False

    suite = Suite()
    pipeline = object()
    task = Task()
    injection = Injection()
    result = _run_case(
        suite,
        pipeline,
        task,
        injection_task=injection,
        injections={"vector": "payload"},
        continue_on_error=False,
    )

    assert suite.called_with == (pipeline, task, injection, {"vector": "payload"})
    assert result.utility is True
    assert result.security is False
    assert result.attack_success is False


def test_pipeline_uses_task_agent_model_and_deepseek_route_without_calling_api() -> None:
    config = ExperimentModelConfig(
        attack_model="attacker",
        defense_model="defender",
        judge_model="judge",
        task_agent_model="deepseek-task",
        base_url="https://openai.example/v1",
        api_key="openai-key",
        deepseek_base_url="https://deepseek.example/v1",
        deepseek_api_key="deepseek-key",
    )

    pipeline = build_dojo_pipeline(config, max_tokens=8192)

    assert pipeline.name == "deepseek-task local"
    from agentdojo.attacks.base_attacks import get_model_name_from_pipeline

    assert get_model_name_from_pipeline(pipeline) == "Local model"
    llm = list(pipeline.elements)[2]
    assert llm.model == "deepseek-task"
    assert llm.max_output_tokens == 8192
    assert str(llm.client.base_url) == "https://deepseek.example/v1/"


def test_pipeline_sends_standard_system_role_to_compatible_gateway() -> None:
    from agentdojo.functions_runtime import EmptyEnv, FunctionsRuntime
    from agentdojo.types import ChatSystemMessage, text_content_block_from_string

    config = ExperimentModelConfig(
        attack_model="attacker",
        defense_model="defender",
        judge_model="judge",
        task_agent_model="deepseek-task",
        base_url="https://openai.example/v1",
        api_key="openai-key",
        deepseek_base_url="https://deepseek.example/v1",
        deepseek_api_key="deepseek-key",
    )
    pipeline = build_dojo_pipeline(config)
    llm = list(pipeline.elements)[2]

    class Completions:
        def __init__(self) -> None:
            self.request = None

        def create(self, **kwargs):
            self.request = kwargs
            message = SimpleNamespace(content="done", tool_calls=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    completions = Completions()
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    system_message = ChatSystemMessage(
        role="system", content=[text_content_block_from_string("system prompt")]
    )

    llm.query(
        "query",
        FunctionsRuntime([]),
        EmptyEnv(),
        messages=[system_message],
    )

    assert completions.request is not None
    assert completions.request["messages"][0]["role"] == "system"


def test_safe_error_includes_transport_cause() -> None:
    try:
        try:
            raise OSError("dns failed")
        except OSError as cause:
            raise RuntimeError("connection failed") from cause
    except RuntimeError as exc:
        error = _safe_error(exc)

    assert error == "RuntimeError: connection failed; caused by OSError: dns failed"


def test_tool_filter_sends_standard_system_role() -> None:
    from agentdojo.functions_runtime import EmptyEnv, FunctionsRuntime
    from agentdojo.types import ChatSystemMessage, text_content_block_from_string

    config = ExperimentModelConfig(
        attack_model="attacker",
        defense_model="defender",
        judge_model="judge",
        task_agent_model="deepseek-task",
        base_url="https://openai.example/v1",
        api_key="openai-key",
        deepseek_base_url="https://deepseek.example/v1",
        deepseek_api_key="deepseek-key",
    )
    pipeline = build_dojo_pipeline(config, defense="tool_filter")
    tool_filter = list(pipeline.elements)[2]

    class Completions:
        def __init__(self) -> None:
            self.request = None

        def create(self, **kwargs):
            self.request = kwargs
            message = SimpleNamespace(content="", tool_calls=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    completions = Completions()
    tool_filter.client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    system_message = ChatSystemMessage(
        role="system", content=[text_content_block_from_string("system prompt")]
    )

    tool_filter.query(
        "query",
        FunctionsRuntime([]),
        EmptyEnv(),
        messages=[system_message],
    )

    assert completions.request is not None
    assert completions.request["messages"][0]["role"] == "system"
