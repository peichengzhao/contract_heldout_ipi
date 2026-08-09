import pytest

from contract_heldout_ipi.agent import (
    ExperimentModelConfig,
    ModelAPIError,
    ModelConfig,
    ModelConfigurationError,
    ModelResponseError,
    OpenAICompatibleChatClient,
)


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

    def post(self, url, *, headers, payload, timeout):
        self.call_count += 1
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
        transport=transport,
    )

    response = client.complete([{"role": "user", "content": "Task"}], [])

    assert response.content == "Done"
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
