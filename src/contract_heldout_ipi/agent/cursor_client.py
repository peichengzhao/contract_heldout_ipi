"""Cursor SDK-backed chat client for text-only attack-model calls."""

from __future__ import annotations

import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .model import (
    ModelAPIError,
    ModelConfigurationError,
    ModelResponse,
    ModelResponseError,
    _compute_backoff_seconds,
    _log_retry_wait,
    _should_retry_again,
)

PromptFn = Callable[[str], str]


class CursorChatClient:
    """Adapter that maps ``ChatModelClient.complete`` onto Cursor Agent prompts.

    Uses ``tools=[]`` so the model can only return text. This is intended for
    attack / synthesis roles that expect JSON text, not IDE tool use.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        cwd: str | Path | None = None,
        timeout: float = 180.0,
        max_retries: int = 2,
        retry_delay: float = 1.0,
        retry_max_delay: float = 120.0,
        retry_budget_seconds: float = 0.0,
        retry_jitter: bool = True,
        prompt_fn: PromptFn | None = None,
    ):
        if not model.strip():
            raise ModelConfigurationError("model must not be empty")
        if not api_key.strip():
            raise ModelConfigurationError("api_key must not be empty")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if retry_delay < 0:
            raise ValueError("retry_delay must be non-negative")
        if retry_max_delay < 0:
            raise ValueError("retry_max_delay must be non-negative")
        if retry_budget_seconds < 0:
            raise ValueError("retry_budget_seconds must be non-negative")
        self.model = model.strip()
        self.api_key = api_key.strip()
        self.cwd = str(cwd) if cwd is not None else None
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_max_delay = retry_max_delay
        self.retry_budget_seconds = retry_budget_seconds
        self.retry_jitter = retry_jitter
        self._prompt_fn = prompt_fn
        self._owned_cwd: tempfile.TemporaryDirectory[str] | None = None
        self._client: Any | None = None
        self._http_client: Any | None = None

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        if tools:
            raise ModelResponseError(
                "CursorChatClient only supports text completion with tools=[]"
            )
        prompt = _format_chat_messages(messages)
        started = time.monotonic()
        deadline = (
            started + self.retry_budget_seconds
            if self.retry_budget_seconds > 0
            else None
        )
        attempt = 0
        content = ""
        while True:
            try:
                content = self._run_prompt(prompt)
                break
            except ModelAPIError as exc:
                safe_error = ModelAPIError(
                    str(exc).replace(self.api_key, "<redacted>"),
                    status_code=exc.status_code,
                    retryable=exc.retryable,
                )
                if not exc.retryable or not _should_retry_again(
                    attempt=attempt,
                    max_retries=self.max_retries,
                    deadline=deadline,
                ):
                    raise safe_error from exc
                delay = _compute_backoff_seconds(
                    attempt,
                    base=self.retry_delay,
                    max_delay=self.retry_max_delay,
                    jitter=self.retry_jitter,
                    deadline=deadline,
                )
                _log_retry_wait(
                    model=self.model,
                    attempt=attempt + 1,
                    delay=delay,
                    error=safe_error,
                    deadline=deadline,
                )
                time.sleep(delay)
                attempt += 1
        if not content.strip():
            raise ModelResponseError("Cursor agent returned empty text")
        return ModelResponse(content=content)

    def close(self) -> None:
        client = self._client
        http_client = self._http_client
        self._client = None
        self._http_client = None
        try:
            if client is not None:
                client.close()
        finally:
            if http_client is not None:
                http_client.close()
            if self._owned_cwd is not None:
                self._owned_cwd.cleanup()
                self._owned_cwd = None

    def __enter__(self) -> CursorChatClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from cursor_sdk import CursorClient, DefaultHttpxClient
        except ImportError as exc:
            raise ModelConfigurationError(
                "cursor-sdk is required for attack_provider=cursor; "
                "install with: pip install cursor-sdk"
            ) from exc

        cwd = self._resolve_cwd()
        # Local bridge listens on 127.0.0.1. If HTTP(S)_PROXY is set, httpx will
        # wrongly proxy those loopback calls and fail with Connection refused.
        self._http_client = DefaultHttpxClient(trust_env=False, timeout=self.timeout)
        self._client = CursorClient.launch_bridge(
            workspace=cwd,
            http_client=self._http_client,
            client_timeout=self.timeout,
        )
        return self._client

    def _resolve_cwd(self) -> str:
        if self.cwd is not None:
            return self.cwd
        if self._owned_cwd is None:
            self._owned_cwd = tempfile.TemporaryDirectory(
                prefix="contract-heldout-ipi-cursor-"
            )
        return self._owned_cwd.name

    def _run_prompt(self, prompt: str) -> str:
        if self._prompt_fn is not None:
            return self._prompt_fn(prompt)
        try:
            from cursor_sdk import Agent, AgentOptions, CursorAgentError, LocalAgentOptions
        except ImportError as exc:
            raise ModelConfigurationError(
                "cursor-sdk is required for attack_provider=cursor; "
                "install with: pip install cursor-sdk"
            ) from exc

        cwd = self._resolve_cwd()
        try:
            client = self._ensure_client()
            result = Agent.prompt(
                prompt,
                AgentOptions(
                    model=self.model,
                    api_key=self.api_key,
                    tools=[],
                    local=LocalAgentOptions(cwd=cwd),
                ),
                client=client,
            )
        except CursorAgentError as exc:
            raise ModelAPIError(
                _cursor_error_message(exc),
                retryable=bool(getattr(exc, "is_retryable", False)),
            ) from exc
        except Exception as exc:  # noqa: BLE001 - normalize provider failures
            message = str(exc).replace(self.api_key, "<redacted>")
            raise ModelAPIError(
                _cursor_error_message(exc, fallback=message),
                retryable=True,
            ) from exc

        status = getattr(result, "status", None)
        if status == "error":
            raise ModelAPIError(
                f"Cursor agent run failed: {getattr(result, 'id', '<unknown>')}",
                retryable=False,
            )
        text = getattr(result, "result", None)
        if not isinstance(text, str):
            raise ModelResponseError("Cursor agent result is not text")
        return text


def _format_chat_messages(messages: list[dict[str, Any]]) -> str:
    if not messages:
        raise ModelResponseError("messages must not be empty")
    parts: list[str] = [
        "Respond with the final answer only. Do not call tools or edit files.",
        "",
    ]
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content")
        if content is None:
            continue
        if not isinstance(content, str):
            raise ModelResponseError("message content must be a string")
        parts.append(f"[{role}]\n{content}")
    return "\n\n".join(parts)


def _cursor_error_message(exc: BaseException, *, fallback: str | None = None) -> str:
    message = fallback or str(exc)
    proxy = (
        os.environ.get("HTTP_PROXY")
        or os.environ.get("http_proxy")
        or os.environ.get("HTTPS_PROXY")
        or os.environ.get("https_proxy")
    )
    if "Connection refused" in message and proxy:
        return (
            f"Cursor local bridge connection failed while HTTP(S)_PROXY={proxy} "
            "was set. The client now bypasses proxies for loopback; if this "
            f"persists, unset the proxy for this shell. Underlying error: {message}"
        )
    if "Connection refused" in message:
        return (
            "Cursor local bridge connection failed (Connection refused). "
            f"Underlying error: {message}"
        )
    return f"Cursor agent failed: {message}"
