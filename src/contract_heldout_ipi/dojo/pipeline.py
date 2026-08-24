"""Build AgentDojo pipelines from the repository's private model config."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from contract_heldout_ipi.agent import ExperimentModelConfig

from .adapter import require_agentdojo


DOJO_DEFENSES = (
    "none",
    "tool_filter",
    "repeat_user_prompt",
    "spotlighting_with_delimiting",
)


def build_dojo_pipeline(
    config: ExperimentModelConfig,
    *,
    defense: str = "none",
    temperature: float = 0.0,
    timeout: float = 180.0,
    max_retries: int = 4,
    max_tokens: int = 4096,
    system_message: str | None = None,
    tool_output_format: str = "yaml",
) -> Any:
    """Create a native AgentDojo tool-use pipeline for ``task_agent_model``."""
    require_agentdojo()
    if defense not in DOJO_DEFENSES:
        raise ValueError(
            f"unknown AgentDojo defense {defense!r}; choose one of: "
            f"{', '.join(DOJO_DEFENSES)}"
        )
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if tool_output_format not in {"yaml", "json"}:
        raise ValueError("tool_output_format must be 'yaml' or 'json'")

    import openai
    from agentdojo.agent_pipeline.agent_pipeline import (
        TOOL_FILTER_PROMPT,
        AgentPipeline,
        PipelineConfig,
    )
    from openai._types import NOT_GIVEN

    from agentdojo.agent_pipeline.llms.openai_llm import (
        OpenAILLM,
        OpenAILLMToolFilter,
        _function_to_openai,
        _message_to_openai,
        _openai_to_assistant_message,
    )
    from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
    from agentdojo.types import (
        ChatMessage,
        ChatUserMessage,
        get_text_content_as_str,
        text_content_block_from_string,
    )

    def compatible_messages(messages: Sequence[ChatMessage], model: str) -> list:
        converted_messages = []
        for message in messages:
            converted = _message_to_openai(message, model)
            # AgentDojo 0.1.35 assumes an OpenAI endpoint that supports the
            # newer `developer` role. DeepSeek and many compatible gateways
            # only implement the standard `system` role.
            if converted.get("role") == "developer":
                converted = {**converted, "role": "system"}
            converted_messages.append(converted)
        return converted_messages

    class BoundedOpenAILLM(OpenAILLM):
        """Pinned AgentDojo OpenAI element with an explicit output budget."""

        def __init__(self, *args: Any, max_output_tokens: int, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.max_output_tokens = max_output_tokens

        def query(
            self,
            query: str,
            runtime: FunctionsRuntime,
            env: Env = EmptyEnv(),
            messages: Sequence[ChatMessage] = (),
            extra_args: dict | None = None,
        ):
            openai_messages = compatible_messages(messages, self.model)
            openai_tools = [
                _function_to_openai(tool) for tool in runtime.functions.values()
            ]
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                tools=openai_tools or NOT_GIVEN,
                tool_choice="auto" if openai_tools else NOT_GIVEN,
                temperature=self.temperature or NOT_GIVEN,
                reasoning_effort=self.reasoning_effort or NOT_GIVEN,
                max_tokens=self.max_output_tokens,
            )
            output = _openai_to_assistant_message(completion.choices[0].message)
            return (
                query,
                runtime,
                env,
                [*messages, output],
                extra_args or {},
            )

    class CompatibleOpenAILLMToolFilter(OpenAILLMToolFilter):
        """AgentDojo tool filter using gateway-compatible message roles."""

        def query(
            self,
            query: str,
            runtime: FunctionsRuntime,
            env: Env = EmptyEnv(),
            messages: Sequence[ChatMessage] = (),
            extra_args: dict | None = None,
        ):
            filter_message = ChatUserMessage(
                role="user",
                content=[text_content_block_from_string(self.prompt)],
            )
            messages = [*messages, filter_message]
            openai_messages = compatible_messages(messages, self.model)
            openai_tools = [
                _function_to_openai(tool) for tool in runtime.functions.values()
            ]
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                tools=openai_tools or NOT_GIVEN,
                tool_choice="none",
                temperature=self.temperature or NOT_GIVEN,
                max_tokens=max_tokens,
            )
            output = _openai_to_assistant_message(completion.choices[0].message)
            selected_tools = {}
            output_text = (
                get_text_content_as_str(output["content"])
                if output["content"] is not None
                else ""
            )
            for tool_name, tool in runtime.functions.items():
                if tool_name in output_text:
                    selected_tools[tool_name] = tool
            runtime.update_functions(selected_tools)
            return query, runtime, env, [*messages, output], extra_args or {}

    model_config = config.for_role("task_agent")
    client = openai.OpenAI(
        api_key=model_config.api_key,
        base_url=model_config.base_url,
        timeout=timeout,
        max_retries=max_retries,
    )
    llm = BoundedOpenAILLM(
        client,
        model_config.model,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    # AgentDojo uses the element name for attack metadata and trace paths.
    llm.name = model_config.model
    pipeline = AgentPipeline.from_config(
        PipelineConfig(
            llm=llm,
            model_id=None,
            defense=None if defense == "none" else defense,
            system_message_name=None,
            system_message=system_message,
            tool_output_format=tool_output_format,
        )
    )
    if defense == "tool_filter":
        elements = list(pipeline.elements)
        elements[2] = CompatibleOpenAILLMToolFilter(
            TOOL_FILTER_PROMPT,
            llm.client,
            llm.model,
            temperature=temperature,
        )
        pipeline.elements = elements
    return pipeline
