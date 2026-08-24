"""Provider-neutral chat model protocol and OpenAI-compatible implementation."""

from __future__ import annotations

import json
import os
import ast
import http.client
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import urlparse


ModelRole = Literal[
    "attack",
    "defense",
    "judge",
    "agent",
    "rule_judge",
    "optimizer",
    "task_agent",
]


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
    """Model-role configuration for attack, defense, judging, and optimization."""

    attack_model: str
    defense_model: str
    judge_model: str
    base_url: str
    api_key: str
    agent_model: str | None = None
    attack_provider: Literal["openai", "cursor"] = "openai"
    cursor_api_key: str | None = None
    rule_judge_model: str | None = None
    optimizer_model: str | None = None
    task_agent_model: str | None = None
    deepseek_base_url: str | None = None
    deepseek_api_key: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("attack_model", "defense_model", "judge_model"):
            if not getattr(self, field_name).strip():
                raise ModelConfigurationError(f"{field_name} must not be empty")
        for field_name in (
            "agent_model",
            "rule_judge_model",
            "optimizer_model",
            "task_agent_model",
            "deepseek_base_url",
            "deepseek_api_key",
        ):
            value = getattr(self, field_name)
            if value is not None and not value.strip():
                raise ModelConfigurationError(
                    f"{field_name} must not be empty when set"
                )
        if self.attack_provider not in {"openai", "cursor"}:
            raise ModelConfigurationError(
                "attack_provider must be 'openai' or 'cursor'"
            )
        if self.attack_provider == "cursor":
            if not self.cursor_api_key or not self.cursor_api_key.strip():
                raise ModelConfigurationError(
                    "cursor_api_key is required when attack_provider=cursor"
                )
        # Reuse URL and credential validation from the single-client config.
        ModelConfig(self.defense_model, self.base_url, self.api_key)
        if (self.deepseek_base_url is None) != (self.deepseek_api_key is None):
            raise ModelConfigurationError(
                "deepseek_base_url and deepseek_api_key must be configured together"
            )
        if self.deepseek_base_url is not None and self.deepseek_api_key is not None:
            ModelConfig(
                "deepseek-route-check",
                self.deepseek_base_url,
                self.deepseek_api_key,
            )

    @classmethod
    def from_file(cls, path: str | Path) -> ExperimentModelConfig:
        values = _read_named_config_fields(
            path,
            {
                "attack_model",
                "defense_model",
                "agent_model",
                "rule_judge_model",
                "optimizer_model",
                "task_agent_model",
                "deepseek_base_url",
                "deepseek_api_key",
                "judge_model",
                "base_url",
                "api_key",
                "attack_provider",
                "cursor_api_key",
                "cur_api_key",
                "cur_attack_model",
                "protect_model",
            },
        )
        # Backward-compatible aliases used by the repository's legacy
        # `model_api.py`. Parsing is static; the Python file is never executed.
        protect_model = values.pop("protect_model", "").strip()
        if protect_model:
            values.setdefault("defense_model", protect_model)
            values.setdefault("judge_model", protect_model)
            values.setdefault("rule_judge_model", protect_model)
            values.setdefault("task_agent_model", protect_model)
        return cls._from_values(values, source="experiment config")

    @classmethod
    def from_env(cls) -> ExperimentModelConfig:
        values = {
            "attack_model": os.getenv("ATTACK_MODEL", ""),
            "defense_model": os.getenv("DEFENSE_MODEL", ""),
            "agent_model": os.getenv("AGENT_MODEL", ""),
            "rule_judge_model": os.getenv("RULE_JUDGE_MODEL", ""),
            "optimizer_model": os.getenv("OPTIMIZER_MODEL", ""),
            "task_agent_model": os.getenv("TASK_AGENT_MODEL", ""),
            "deepseek_base_url": os.getenv("DEEPSEEK_BASE_URL", ""),
            "deepseek_api_key": os.getenv("DEEPSEEK_API_KEY", ""),
            "judge_model": os.getenv("JUDGE_MODEL", ""),
            "base_url": os.getenv("OPENAI_BASE_URL", ""),
            "api_key": os.getenv("OPENAI_API_KEY", ""),
            "attack_provider": os.getenv("ATTACK_PROVIDER", ""),
            "cursor_api_key": os.getenv("CURSOR_API_KEY", ""),
            "cur_attack_model": os.getenv("CUR_ATTACK_MODEL", ""),
        }
        # Empty optional fields should fall back inside `_from_values`.
        for optional in (
            "defense_model",
            "agent_model",
            "rule_judge_model",
            "optimizer_model",
            "task_agent_model",
            "deepseek_base_url",
            "deepseek_api_key",
            "attack_provider",
            "cursor_api_key",
            "cur_attack_model",
        ):
            if not values.get(optional):
                values.pop(optional, None)
        return cls._from_values(values, source="environment experiment configuration")

    @classmethod
    def _from_values(
        cls, values: dict[str, str], *, source: str
    ) -> ExperimentModelConfig:
        cursor_api_key = (
            values.get("cursor_api_key", "").strip()
            or values.get("cur_api_key", "").strip()
            or os.getenv("CURSOR_API_KEY", "").strip()
            or None
        )
        attack_provider_raw = values.get("attack_provider", "").strip().lower()
        cur_attack_model = values.get("cur_attack_model", "").strip()
        if not attack_provider_raw:
            attack_provider_raw = "cursor" if cur_attack_model else "openai"
        if attack_provider_raw not in {"openai", "cursor"}:
            raise ModelConfigurationError(
                f"{source} has invalid attack_provider: {attack_provider_raw}"
            )
        attack_provider: Literal["openai", "cursor"] = (
            "cursor" if attack_provider_raw == "cursor" else "openai"
        )

        if attack_provider == "cursor":
            attack_model = (
                cur_attack_model or values.get("attack_model", "")
            ).strip()
            if not attack_model:
                raise ModelConfigurationError(
                    f"{source} is missing: attack_model (or cur_attack_model)"
                )
            if not values.get("defense_model", "").strip():
                raise ModelConfigurationError(
                    f"{source} requires defense_model when attack_provider=cursor"
                )
        else:
            attack_model = values.get("attack_model", "").strip()

        required = {"judge_model", "base_url", "api_key"}
        if attack_provider == "openai":
            required.add("attack_model")
        present = {
            key
            for key, value in {
                **values,
                "attack_model": attack_model,
            }.items()
            if value and str(value).strip()
        }
        missing = required - present
        if missing:
            names = ", ".join(sorted(missing))
            raise ModelConfigurationError(f"{source} is missing: {names}")

        defense_model = values.get("defense_model", "").strip() or attack_model
        agent_raw = values.get("agent_model", "").strip()
        rule_judge_raw = values.get("rule_judge_model", "").strip()
        optimizer_raw = values.get("optimizer_model", "").strip()
        task_agent_raw = values.get("task_agent_model", "").strip()
        deepseek_base_url = values.get("deepseek_base_url", "").strip()
        deepseek_api_key = values.get("deepseek_api_key", "").strip()
        return cls(
            attack_model=attack_model,
            defense_model=defense_model,
            judge_model=values["judge_model"].strip(),
            base_url=values["base_url"].strip(),
            api_key=values["api_key"].strip(),
            agent_model=agent_raw or None,
            rule_judge_model=rule_judge_raw or None,
            optimizer_model=optimizer_raw or None,
            task_agent_model=task_agent_raw or None,
            deepseek_base_url=deepseek_base_url or None,
            deepseek_api_key=deepseek_api_key or None,
            attack_provider=attack_provider,
            cursor_api_key=cursor_api_key,
        )

    def provider_for_role(self, role: ModelRole) -> Literal["openai", "cursor"]:
        if role == "attack" and self.attack_provider == "cursor":
            return "cursor"
        return "openai"

    def for_role(self, role: ModelRole) -> ModelConfig:
        if self.provider_for_role(role) == "cursor":
            raise ModelConfigurationError(
                "attack role uses Cursor; call build_role_client() instead of for_role()"
            )
        model = {
            "attack": self.attack_model,
            "defense": self.defense_model,
            "judge": self.judge_model,
            "rule_judge": self.rule_judge_model or self.judge_model,
            "optimizer": self.optimizer_model or self.judge_model,
            # `agent` remains a backward-compatible alias for task_agent.
            "agent": self.task_agent_model or self.agent_model or self.defense_model,
            "task_agent": (
                self.task_agent_model or self.agent_model or self.defense_model
            ),
        }[role]
        use_deepseek_route = (
            model.casefold().startswith("deepseek")
            and self.deepseek_base_url is not None
            and self.deepseek_api_key is not None
        )
        return ModelConfig(
            model=model,
            base_url=self.deepseek_base_url if use_deepseek_route else self.base_url,
            api_key=self.deepseek_api_key if use_deepseek_route else self.api_key,
        )


def build_role_client(
    config: ExperimentModelConfig,
    role: ModelRole,
    *,
    timeout: float = 180.0,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    max_retries: int = 2,
    retry_delay: float = 1.0,
) -> ChatModelClient:
    """Build the chat client for a model role.

    Attack may use the Cursor SDK (``attack_provider=cursor``); other roles use
    the OpenAI-compatible gateway.
    """
    if config.provider_for_role(role) == "cursor":
        from .cursor_client import CursorChatClient

        assert config.cursor_api_key is not None
        return CursorChatClient(
            model=config.attack_model,
            api_key=config.cursor_api_key,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
    return OpenAICompatibleChatClient(
        config.for_role(role),
        timeout=timeout,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )


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
        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
            http.client.HTTPException,
        ) as exc:
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
                return _parse_chat_completion(response)
            except ModelAPIError as exc:
                safe_error = ModelAPIError(
                    str(exc).replace(self.config.api_key, "<redacted>"),
                    status_code=exc.status_code,
                    retryable=exc.retryable,
                )
                if not exc.retryable or attempt >= self.max_retries:
                    raise safe_error from exc
                time.sleep(self.retry_delay * (2**attempt))
            except ModelResponseError as exc:
                contextual = ModelResponseError(
                    f"model '{self.config.model}' returned an invalid response: {exc}"
                )
                if attempt >= self.max_retries:
                    raise contextual from exc
                if "finish_reason='length'" in str(exc):
                    current_limit = payload.get("max_tokens", self.max_tokens)
                    if isinstance(current_limit, int):
                        payload["max_tokens"] = min(current_limit * 2, 65_536)
                time.sleep(self.retry_delay * (2**attempt))
        raise AssertionError("unreachable completion retry state")


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
        first_choice = choices[0] if isinstance(choices[0], dict) else {}
        finish_reason = first_choice.get("finish_reason")
        reasoning = message.get("reasoning_content")
        reasoning_chars = len(reasoning) if isinstance(reasoning, str) else 0
        fields = ",".join(sorted(str(key) for key in message.keys()))
        raise ModelResponseError(
            "assistant returned neither a tool call nor text "
            f"(finish_reason={finish_reason!r}, reasoning_chars={reasoning_chars}, "
            f"message_fields={fields})"
        )
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
    except (
        json.JSONDecodeError,
        UnicodeDecodeError,
        OSError,
        http.client.HTTPException,
    ):
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
    key_aliases = {
        "openai_base_url": "base_url",
        "openai_api_key": "api_key",
    }
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("<!--") or stripped.startswith("#"):
            continue
        if "=" in stripped:
            key, raw_value = stripped.split("=", 1)
            try:
                parsed = ast.literal_eval(raw_value.strip())
            except (SyntaxError, ValueError):
                continue
            if not isinstance(parsed, str):
                continue
            value = parsed
        elif ":" in stripped:
            key, raw_value = stripped.split(":", 1)
            value = raw_value.strip()
        else:
            continue
        normalized_key = key_aliases.get(
            key.strip().lower(), key.strip().lower()
        )
        if normalized_key in field_names:
            values[normalized_key] = value.strip()
    return values
