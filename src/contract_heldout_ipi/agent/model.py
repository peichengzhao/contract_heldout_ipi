"""Provider-neutral chat model protocol and OpenAI-compatible implementation."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import urlparse


class ModelClientError(RuntimeError):
    """Base class for model configuration, transport, and response errors."""


class ModelConfigurationError(ModelClientError):
    """Raised when required model connection settings are missing or invalid."""


class ModelAPIError(ModelClientError):
    """Raised when the remote API rejects a request or cannot be reached."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class ModelResponseError(ModelClientError):
    """Raised when a model response cannot be converted into one agent action."""


@dataclass(frozen=True)
class ModelConfig:
    model: str
    base_url: str
    api_key: str

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ModelConfigurationError("model must not be empty")
        parsed_url = urlparse(self.base_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ModelConfigurationError("base_url must be an absolute HTTP(S) URL")
        if not self.api_key.strip():
            raise ModelConfigurationError("api_key must not be empty")

    @classmethod
    def from_file(cls, path: str | Path) -> ModelConfig:
        """Read exact ``model``, ``base_url``, and ``api_key`` fields.

        Other prose or unnamed values in a Markdown configuration file are ignored.
        """
        values = _read_named_config_fields(
            path, {"model", "base_url", "api_key"}
        )
        missing = {"model", "base_url", "api_key"} - values.keys()
        if missing:
            names = ", ".join(sorted(missing))
            raise ModelConfigurationError(f"model config is missing: {names}")
        return cls(**values)

    @classmethod
    def from_env(cls) -> ModelConfig:
        values = {
            "model": os.getenv("OPENAI_MODEL", ""),
            "base_url": os.getenv("OPENAI_BASE_URL", ""),
            "api_key": os.getenv("OPENAI_API_KEY", ""),
        }
        missing = [key for key, value in values.items() if not value]
        if missing:
            names = ", ".join(sorted(missing))
            raise ModelConfigurationError(
                f"missing environment model configuration: {names}"
            )
        return cls(**values)


@dataclass(frozen=True)
class ExperimentModelConfig:
    """Model-role configuration for dynamic attack/defense/judge evaluation."""

    attack_model: str
    defense_model: str
    judge_model: str
    base_url: str
    api_key: str
    agent_model: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("attack_model", "defense_model", "judge_model"):
            if not getattr(self, field_name).strip():
                raise ModelConfigurationError(f"{field_name} must not be empty")
        if self.agent_model is not None and not self.agent_model.strip():
            raise ModelConfigurationError("agent_model must not be empty when set")
        # Reuse URL and credential validation from the single-client config.
        ModelConfig(self.defense_model, self.base_url, self.api_key)

    @classmethod
    def from_file(cls, path: str | Path) -> ExperimentModelConfig:
        values = _read_named_config_fields(
            path,
            {
                "attack_model",
                "defense_model",
                "agent_model",
                "judge_model",
                "base_url",
                "api_key",
            },
        )
        return cls._from_values(values, source="experiment config")

    @classmethod
    def from_env(cls) -> ExperimentModelConfig:
        values = {
            "attack_model": os.getenv("ATTACK_MODEL", ""),
            "defense_model": os.getenv("DEFENSE_MODEL", ""),
            "agent_model": os.getenv("AGENT_MODEL", ""),
            "judge_model": os.getenv("JUDGE_MODEL", ""),
            "base_url": os.getenv("OPENAI_BASE_URL", ""),
            "api_key": os.getenv("OPENAI_API_KEY", ""),
        }
        # Empty optional fields should fall back inside `_from_values`.
        if not values.get("defense_model"):
            values.pop("defense_model", None)
        if not values.get("agent_model"):
            values.pop("agent_model", None)
        return cls._from_values(values, source="environment experiment configuration")

    @classmethod
    def _from_values(
        cls, values: dict[str, str], *, source: str
    ) -> ExperimentModelConfig:
        required = {"attack_model", "judge_model", "base_url", "api_key"}
        missing = required - {
            key for key, value in values.items() if value and value.strip()
        }
        if missing:
            names = ", ".join(sorted(missing))
            raise ModelConfigurationError(f"{source} is missing: {names}")
        attack_model = values["attack_model"].strip()
        defense_model = values.get("defense_model", "").strip() or attack_model
        agent_raw = values.get("agent_model", "").strip()
        return cls(
            attack_model=attack_model,
            defense_model=defense_model,
            judge_model=values["judge_model"].strip(),
            base_url=values["base_url"].strip(),
            api_key=values["api_key"].strip(),
            agent_model=agent_raw or None,
        )

    def for_role(
        self, role: Literal["attack", "defense", "judge", "agent"]
    ) -> ModelConfig:
        model = {
            "attack": self.attack_model,
            "defense": self.defense_model,
            "judge": self.judge_model,
            # Task agent may be configured separately when defenses are skipped.
            "agent": self.agent_model or self.defense_model,
        }[role]
        return ModelConfig(model=model, base_url=self.base_url, api_key=self.api_key)


@dataclass(frozen=True)
class ModelToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ModelResponse:
    content: str | None = None
    tool_call: ModelToolCall | None = None
    tool_calls: tuple[ModelToolCall, ...] = ()

    @property
    def all_tool_calls(self) -> tuple[ModelToolCall, ...]:
        if self.tool_call is not None and self.tool_calls:
            raise ModelResponseError(
                "model response cannot set both tool_call and tool_calls"
            )
        if self.tool_call is not None:
            return (self.tool_call,)
        return self.tool_calls


class ChatModelClient(Protocol):
    """Small model boundary consumed by ``LLMAgent`` and faked in tests."""

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        ...


class JsonTransport(Protocol):
    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        ...


class UrllibJsonTransport:
    """Dependency-free JSON-over-HTTP transport."""

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.load(response)
        except urllib.error.HTTPError as exc:
            message = _read_api_error(exc)
            raise ModelAPIError(
                f"model API returned HTTP {exc.code}: {message}",
                status_code=exc.code,
                retryable=exc.code in {408, 409, 429, 500, 502, 503, 504},
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ModelAPIError(
                f"model API connection failed: {exc}", retryable=True
            ) from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ModelResponseError("model API returned invalid JSON") from exc

        if not isinstance(body, dict):
            raise ModelResponseError("model API response must be a JSON object")
        return body


class OpenAICompatibleChatClient:
    """OpenAI Chat Completions-compatible client with injectable transport."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        timeout: float = 180.0,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        max_retries: int = 2,
        retry_delay: float = 1.0,
        transport: JsonTransport | None = None,
    ):
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if retry_delay < 0:
            raise ValueError("retry_delay must be non-negative")
        self.config = config
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.transport = transport or UrllibJsonTransport()

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        for attempt in range(self.max_retries + 1):
            try:
                response = self.transport.post(
                    f"{self.config.base_url.rstrip('/')}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.config.api_key}",
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                        # Some compatible gateways reject urllib's default UA.
                        "User-Agent": "contract-heldout-ipi/0.1",
                    },
                    payload=payload,
                    timeout=self.timeout,
                )
                break
            except ModelAPIError as exc:
                safe_error = ModelAPIError(
                    str(exc).replace(self.config.api_key, "<redacted>"),
                    status_code=exc.status_code,
                    retryable=exc.retryable,
                )
                if not exc.retryable or attempt >= self.max_retries:
                    raise safe_error from exc
                time.sleep(self.retry_delay * (2**attempt))
        return _parse_chat_completion(response)


def _parse_chat_completion(response: dict[str, Any]) -> ModelResponse:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ModelResponseError("chat completion has no choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise ModelResponseError("chat completion has no assistant message")

    tool_calls = message.get("tool_calls") or []
    if not isinstance(tool_calls, list):
        raise ModelResponseError("assistant tool_calls must be a list")
    if tool_calls:
        parsed_calls = tuple(_parse_tool_call(call) for call in tool_calls)
        if len(parsed_calls) == 1:
            return ModelResponse(
                content=_optional_string(message.get("content")),
                tool_call=parsed_calls[0],
            )
        return ModelResponse(
            content=_optional_string(message.get("content")),
            tool_calls=parsed_calls,
        )

    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ModelResponseError("assistant returned neither a tool call nor text")
    return ModelResponse(content=content)


def _parse_tool_call(raw_call: Any) -> ModelToolCall:
    if not isinstance(raw_call, dict):
        raise ModelResponseError("assistant tool call must be an object")
    function = raw_call.get("function")
    if not isinstance(function, dict):
        raise ModelResponseError("assistant tool call has no function")
    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise ModelResponseError("assistant tool call has no function name")

    raw_arguments = function.get("arguments", "{}")
    if isinstance(raw_arguments, str):
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise ModelResponseError("tool-call arguments are not valid JSON") from exc
    else:
        arguments = raw_arguments
    if not isinstance(arguments, dict):
        raise ModelResponseError("tool-call arguments must decode to an object")

    call_id = raw_call.get("id")
    if not isinstance(call_id, str) or not call_id:
        call_id = f"call_{uuid.uuid4().hex}"
    return ModelToolCall(id=call_id, name=name, arguments=arguments)


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _read_api_error(exc: urllib.error.HTTPError) -> str:
    try:
        body = json.loads(exc.read().decode("utf-8", "replace"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return "request rejected"
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return error["message"][:500]
    return "request rejected"


def _read_named_config_fields(
    path: str | Path, field_names: set[str]
) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("<!--") or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        normalized_key = key.strip().lower()
        if normalized_key in field_names:
            values[normalized_key] = value.strip()
    return values
