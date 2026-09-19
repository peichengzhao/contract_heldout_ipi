import http.client

import pytest

from contract_heldout_ipi.agent import (
    CursorChatClient,
    ExperimentModelConfig,
    ModelAPIError,
    ModelConfig,
    ModelConfigurationError,
    ModelResponseError,
    OpenAICompatibleChatClient,
    build_role_client,
)
from contract_heldout_ipi.agent.model import UrllibJsonTransport


def test_urllib_transport_normalizes_incomplete_http_response(monkeypatch):
    def broken_urlopen(request, timeout):
        raise http.client.IncompleteRead(b"", 100)

    monkeypatch.setattr("urllib.request.urlopen", broken_urlopen)

    with pytest.raises(ModelAPIError) as exc_info:
        UrllibJsonTransport().post(
            "https://gateway.example/v1/chat/completions",
            headers={"Authorization": "Bearer test"},
            payload={"model": "test"},
            timeout=30.0,
        )

    assert exc_info.value.retryable is True
    assert "IncompleteRead" in str(exc_info.value)


class StubTransport:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, url, *, headers, payload, timeout):
        self.calls.append(
            {
                "url": url,
                "headers": headers,
                "payload": payload,
                "timeout": timeout,
            }
        )
        return self.response


class SequenceTransport:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.call_count = 0
        self.max_tokens_seen = []

    def post(self, url, *, headers, payload, timeout):
        self.call_count += 1
        self.max_tokens_seen.append(payload.get("max_tokens"))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_model_config_reads_only_explicit_named_fields(tmp_path):
    path = tmp_path / "model.md"
    path.write_text(
        "ignored raw value\n"
        "model: test-model\n"
        "base_url: https://gateway.example/v1\n"
        "api_key: test-secret\n",
        encoding="utf-8",
    )

    config = ModelConfig.from_file(path)

    assert config == ModelConfig(
        model="test-model",
        base_url="https://gateway.example/v1",
        api_key="test-secret",
    )


def test_model_config_rejects_missing_or_invalid_settings(tmp_path):
    path = tmp_path / "model.md"
    path.write_text("model: test-model\napi_key: secret\n", encoding="utf-8")

    with pytest.raises(ModelConfigurationError, match="base_url"):
        ModelConfig.from_file(path)

    with pytest.raises(ModelConfigurationError, match="absolute"):
        ModelConfig("test-model", "not-a-url", "secret")


def test_model_config_reads_single_model_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")

    assert ModelConfig.from_env() == ModelConfig(
        model="test-model",
        base_url="https://gateway.example/v1",
        api_key="test-secret",
    )


def test_experiment_model_config_resolves_three_roles(tmp_path):
    path = tmp_path / "models.md"
    path.write_text(
        "attack_model: attacker\n"
        "defense_model: defender\n"
        "judge_model: judge\n"
        "base_url: https://gateway.example/v1\n"
        "api_key: test-secret\n",
        encoding="utf-8",
    )

    config = ExperimentModelConfig.from_file(path)

    assert config.for_role("attack").model == "attacker"
    assert config.for_role("defense").model == "defender"
    assert config.for_role("judge").model == "judge"
    assert config.for_role("judge").api_key == "test-secret"
    assert config.for_role("rule_judge").model == "judge"
    # Defense agent owns harness updates when OPTIMIZER_MODEL is unset.
    assert config.for_role("optimizer").model == "defender"
    assert config.for_role("task_agent").model == "defender"


def test_experiment_model_config_accepts_work_agent_alias(tmp_path):
    path = tmp_path / "models.py"
    path.write_text(
        'OPENAI_BASE_URL = "https://gateway.example/v1"\n'
        'OPENAI_API_KEY = "test-secret"\n'
        'ATTACK_MODEL = "atk"\n'
        'DEFENSE_MODEL = "defn"\n'
        'JUDGE_MODEL = "jdg"\n'
        'WORK_AGENT = "worker"\n',
        encoding="utf-8",
    )
    config = ExperimentModelConfig.from_file(path)
    assert config.for_role("attack").model == "atk"
    assert config.for_role("defense").model == "defn"
    assert config.for_role("judge").model == "jdg"
    assert config.for_role("rule_judge").model == "jdg"
    assert config.for_role("optimizer").model == "defn"
    assert config.for_role("task_agent").model == "worker"


def test_experiment_model_config_resolves_independent_game_roles(tmp_path):
    path = tmp_path / "models.md"
    path.write_text(
        "attack_model: attacker\n"
        "defense_model: defender-negotiator\n"
        "judge_model: utility-judge\n"
        "rule_judge_model: rule-judge\n"
        "optimizer_model: harness-optimizer\n"
        "task_agent_model: task-agent\n"
        "base_url: https://gateway.example/v1\n"
        "api_key: test-secret\n",
        encoding="utf-8",
    )

    config = ExperimentModelConfig.from_file(path)

    assert config.for_role("defense").model == "defender-negotiator"
    assert config.for_role("judge").model == "utility-judge"
    assert config.for_role("rule_judge").model == "rule-judge"
    assert config.for_role("optimizer").model == "harness-optimizer"
    assert config.for_role("task_agent").model == "task-agent"
    assert config.for_role("agent").model == "task-agent"
    assert build_role_client(config, "rule_judge").config.model == "rule-judge"
    assert build_role_client(config, "optimizer").config.model == "harness-optimizer"
    assert build_role_client(config, "task_agent").config.model == "task-agent"


def test_experiment_model_config_reads_legacy_python_constants(tmp_path):
    path = tmp_path / "model_api.py"
    path.write_text(
        'ATTACK_MODEL = "attacker"\n'
        'OPENAI_BASE_URL = "https://gateway.example/v1"\n'
        'OPENAI_API_KEY = "test-secret"\n'
        'OPTIMIZER_MODEL = "optimizer"\n'
        'PROTECT_MODEL = "protector"\n'
        'llm = make_client(model="ignored")\n',
        encoding="utf-8",
    )

    config = ExperimentModelConfig.from_file(path)

    assert config.for_role("attack").model == "attacker"
    assert config.for_role("defense").model == "protector"
    assert config.for_role("judge").model == "protector"
    assert config.for_role("rule_judge").model == "protector"
    assert config.for_role("task_agent").model == "protector"
    assert config.for_role("optimizer").model == "optimizer"


def test_experiment_model_config_routes_deepseek_separately(tmp_path):
    path = tmp_path / "model_api.py"
    path.write_text(
        'OPENAI_BASE_URL = "https://openai-gateway.example/v1"\n'
        'OPENAI_API_KEY = "openai-secret"\n'
        'DEEPSEEK_BASE_URL = "https://deepseek-gateway.example/v1"\n'
        'DEEPSEEK_API_KEY = "deepseek-secret"\n'
        'ATTACK_MODEL = "deepseek-v4-flash"\n'
        'DEFENSE_MODEL = "deepseek-v4-flash"\n'
        'TASK_AGENT_MODEL = "deepseek-v4-flash"\n'
        'JUDGE_MODEL = "gpt-5.6-sol"\n'
        'RULE_JUDGE_MODEL = "gpt-5.6-sol"\n'
        'OPTIMIZER_MODEL = "deepseek-v4-flash"\n',
        encoding="utf-8",
    )

    config = ExperimentModelConfig.from_file(path)

    assert config.for_role("attack").base_url == "https://deepseek-gateway.example/v1"
    assert config.for_role("attack").api_key == "deepseek-secret"
    assert config.for_role("task_agent").base_url == (
        "https://deepseek-gateway.example/v1"
    )
    assert config.for_role("optimizer").base_url == (
        "https://deepseek-gateway.example/v1"
    )
    assert config.for_role("judge").base_url == "https://openai-gateway.example/v1"
    assert config.for_role("judge").api_key == "openai-secret"
    assert config.for_role("rule_judge").base_url == (
        "https://openai-gateway.example/v1"
    )


def test_experiment_model_config_allows_missing_defense_model(tmp_path):
    path = tmp_path / "models.md"
    path.write_text(
        "<!-- defense_model: ignored -->\n"
        "attack_model: attacker\n"
        "judge_model: judge\n"
        "base_url: https://gateway.example/v1\n"
        "api_key: test-secret\n",
        encoding="utf-8",
    )

    config = ExperimentModelConfig.from_file(path)

    assert config.attack_model == "attacker"
    assert config.defense_model == "attacker"
    assert config.judge_model == "judge"
    assert config.for_role("agent").model == "attacker"


def test_experiment_model_config_uses_optional_agent_model(tmp_path):
    path = tmp_path / "models.md"
    path.write_text(
        "attack_model: attacker\n"
        "agent_model: task-agent\n"
        "judge_model: judge\n"
        "base_url: https://gateway.example/v1\n"
        "api_key: test-secret\n",
        encoding="utf-8",
    )

    config = ExperimentModelConfig.from_file(path)

    assert config.for_role("attack").model == "attacker"
    assert config.for_role("agent").model == "task-agent"
    assert config.for_role("defense").model == "attacker"


def test_experiment_model_config_supports_cursor_attack_provider(tmp_path):
    path = tmp_path / "models.md"
    path.write_text(
        "attack_provider: cursor\n"
        "attack_model: grok-4.5\n"
        "cursor_api_key: crsr_test\n"
        "defense_model: defender\n"
        "judge_model: judge\n"
        "base_url: https://gateway.example/v1\n"
        "api_key: test-secret\n",
        encoding="utf-8",
    )

    config = ExperimentModelConfig.from_file(path)

    assert config.attack_provider == "cursor"
    assert config.attack_model == "grok-4.5"
    assert config.cursor_api_key == "crsr_test"
    assert config.provider_for_role("attack") == "cursor"
    assert config.provider_for_role("judge") == "openai"
    with pytest.raises(ModelConfigurationError, match="build_role_client"):
        config.for_role("attack")
    assert config.for_role("judge").model == "judge"


def test_experiment_model_config_accepts_cur_aliases(tmp_path):
    path = tmp_path / "models.md"
    path.write_text(
        "cur_attack_model: grok-4.5\n"
        "cur_api_key: crsr_alias\n"
        "defense_model: defender\n"
        "judge_model: judge\n"
        "base_url: https://gateway.example/v1\n"
        "api_key: test-secret\n",
        encoding="utf-8",
    )

    config = ExperimentModelConfig.from_file(path)

    assert config.attack_provider == "cursor"
    assert config.attack_model == "grok-4.5"
    assert config.cursor_api_key == "crsr_alias"


def test_cursor_attack_provider_requires_defense_model(tmp_path):
    path = tmp_path / "models.md"
    path.write_text(
        "attack_provider: cursor\n"
        "attack_model: grok-4.5\n"
        "cursor_api_key: crsr_test\n"
        "judge_model: judge\n"
        "base_url: https://gateway.example/v1\n"
        "api_key: test-secret\n",
        encoding="utf-8",
    )

    with pytest.raises(ModelConfigurationError, match="defense_model"):
        ExperimentModelConfig.from_file(path)


def test_build_role_client_uses_cursor_for_attack():
    config = ExperimentModelConfig(
        attack_model="grok-4.5",
        defense_model="defender",
        judge_model="judge",
        base_url="https://gateway.example/v1",
        api_key="test-secret",
        attack_provider="cursor",
        cursor_api_key="crsr_test",
    )

    attack = build_role_client(config, "attack")
    judge = build_role_client(config, "judge")

    assert isinstance(attack, CursorChatClient)
    assert attack.model == "grok-4.5"
    assert isinstance(judge, OpenAICompatibleChatClient)


def test_cursor_chat_client_returns_text_via_injected_prompt_fn():
    client = CursorChatClient(
        model="grok-4.5",
        api_key="crsr_test",
        prompt_fn=lambda prompt: '{"payload": "injected"}',
    )

    response = client.complete(
        [
            {"role": "system", "content": "Return JSON"},
            {"role": "user", "content": "Generate"},
        ],
        tools=[],
    )

    assert response.content == '{"payload": "injected"}'


def test_cursor_chat_client_rejects_tools():
    client = CursorChatClient(
        model="grok-4.5",
        api_key="crsr_test",
        prompt_fn=lambda prompt: "ok",
    )

    with pytest.raises(ModelResponseError, match="tools=\\[\\]"):
        client.complete([{"role": "user", "content": "hi"}], [{"type": "function"}])


def test_openai_compatible_client_parses_one_tool_call():
    transport = StubTransport(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "read_email",
                                    "arguments": '{"email_id": "e1"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
    )
    client = OpenAICompatibleChatClient(
        ModelConfig("test-model", "https://gateway.example/v1", "secret"),
        timeout=12,
        max_tokens=50,
        transport=transport,
    )

    response = client.complete(
        [{"role": "user", "content": "Read my email"}],
        [{"type": "function", "function": {"name": "read_email"}}],
    )

    assert response.tool_call.name == "read_email"
    assert response.tool_call.arguments == {"email_id": "e1"}
    request = transport.calls[0]
    assert request["url"] == "https://gateway.example/v1/chat/completions"
    assert request["headers"]["Authorization"] == "Bearer secret"
    assert request["headers"]["Accept"] == "application/json"
    assert request["headers"]["User-Agent"] == "contract-heldout-ipi/0.1"
    assert request["payload"]["model"] == "test-model"
    assert request["payload"]["tool_choice"] == "auto"


def test_openai_compatible_client_parses_final_text():
    transport = StubTransport(
        {"choices": [{"message": {"role": "assistant", "content": "Done"}}]}
    )
    client = OpenAICompatibleChatClient(
        ModelConfig("test-model", "https://gateway.example/v1", "secret"),
        transport=transport,
    )

    response = client.complete([{"role": "user", "content": "Task"}], [])

    assert response.content == "Done"
    assert response.tool_call is None
    assert "tools" not in transport.calls[0]["payload"]


def test_multiple_tool_calls_are_preserved_in_response_order():
    first_call = {
        "id": "call-1",
        "type": "function",
        "function": {"name": "list_emails", "arguments": "{}"},
    }
    second_call = {
        "id": "call-2",
        "type": "function",
        "function": {
            "name": "read_email",
            "arguments": '{"email_id": "e1"}',
        },
    }
    transport = StubTransport(
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [first_call, second_call],
                    }
                }
            ]
        }
    )
    client = OpenAICompatibleChatClient(
        ModelConfig("test-model", "https://gateway.example/v1", "secret"),
        transport=transport,
    )

    response = client.complete([], [])

    assert [call.name for call in response.all_tool_calls] == [
        "list_emails",
        "read_email",
    ]


def test_retryable_model_error_is_retried():
    transport = SequenceTransport(
        [
            ModelAPIError("read timed out", retryable=True),
            {"choices": [{"message": {"content": "Done"}}]},
        ]
    )
    client = OpenAICompatibleChatClient(
        ModelConfig("test-model", "https://gateway.example/v1", "secret"),
        max_retries=2,
        retry_delay=0,
        retry_jitter=False,
        transport=transport,
    )

    response = client.complete([{"role": "user", "content": "Task"}], [])

    assert response.content == "Done"
    assert transport.call_count == 2


def test_exponential_backoff_caps_and_respects_budget(monkeypatch):
    from contract_heldout_ipi.agent import model as model_mod

    sleeps: list[float] = []
    monkeypatch.setattr(model_mod.time, "sleep", lambda s: sleeps.append(s))
    # Freeze monotonic progression except we advance manually via sleeps side effect.
    clock = {"t": 1000.0}

    def fake_monotonic() -> float:
        return clock["t"]

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock["t"] += seconds

    monkeypatch.setattr(model_mod.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(model_mod.time, "sleep", fake_sleep)

    transport = SequenceTransport(
        [
            ModelAPIError("503 channel", status_code=503, retryable=True),
            ModelAPIError("503 channel", status_code=503, retryable=True),
            ModelAPIError("503 channel", status_code=503, retryable=True),
            {"choices": [{"message": {"content": "Recovered"}}]},
        ]
    )
    client = OpenAICompatibleChatClient(
        ModelConfig("test-model", "https://gateway.example/v1", "secret"),
        max_retries=1,  # would stop after 1 retry without budget
        retry_delay=2.0,
        retry_max_delay=8.0,
        retry_budget_seconds=60.0,
        retry_jitter=False,
        transport=transport,
    )

    response = client.complete([{"role": "user", "content": "Task"}], [])

    assert response.content == "Recovered"
    assert transport.call_count == 4
    # attempt 0 -> 2, attempt 1 -> 4, attempt 2 -> 8 (capped)
    assert sleeps == [2.0, 4.0, 8.0]


def test_compute_backoff_seconds_caps():
    from contract_heldout_ipi.agent.model import _compute_backoff_seconds

    assert _compute_backoff_seconds(
        0, base=2.0, max_delay=100.0, jitter=False, deadline=None
    ) == 2.0
    assert _compute_backoff_seconds(
        1, base=2.0, max_delay=100.0, jitter=False, deadline=None
    ) == 4.0
    assert _compute_backoff_seconds(
        10, base=2.0, max_delay=30.0, jitter=False, deadline=None
    ) == 30.0


def test_empty_assistant_response_is_retried_with_model_context():
    empty = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {"content": None, "reasoning_content": "thinking"},
            }
        ]
    }
    transport = SequenceTransport(
        [empty, {"choices": [{"message": {"content": "Done"}}]}]
    )
    client = OpenAICompatibleChatClient(
        ModelConfig("test-model", "https://gateway.example/v1", "secret"),
        max_retries=1,
        retry_delay=0,
        transport=transport,
    )

    response = client.complete([{"role": "user", "content": "Task"}], [])

    assert response.content == "Done"
    assert transport.call_count == 2
    assert transport.max_tokens_seen == [1024, 2048]


def test_repeated_empty_response_reports_model_and_finish_reason():
    empty = {
        "choices": [
            {"finish_reason": "length", "message": {"content": None}}
        ]
    }
    transport = SequenceTransport([empty, empty])
    client = OpenAICompatibleChatClient(
        ModelConfig("test-model", "https://gateway.example/v1", "secret"),
        max_retries=1,
        retry_delay=0,
        transport=transport,
    )

    with pytest.raises(
        ModelResponseError, match="model 'test-model'.*finish_reason='length'"
    ):
        client.complete([{"role": "user", "content": "Task"}], [])

    assert transport.call_count == 2


def test_permanent_model_error_is_not_retried():
    transport = SequenceTransport(
        [ModelAPIError("forbidden", status_code=403, retryable=False)]
    )
    client = OpenAICompatibleChatClient(
        ModelConfig("test-model", "https://gateway.example/v1", "secret"),
        max_retries=2,
        retry_delay=0,
        transport=transport,
    )

    with pytest.raises(ModelAPIError, match="forbidden"):
        client.complete([{"role": "user", "content": "Task"}], [])

    assert transport.call_count == 1
