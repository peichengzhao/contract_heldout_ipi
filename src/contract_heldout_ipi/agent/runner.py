"""Execution loop connecting an agent policy to an episode sandbox."""

from __future__ import annotations

from typing import Any

from contract_heldout_ipi.contract.models import EpisodeContract
from contract_heldout_ipi.env import EmailSandbox, ToolBudgetExceeded
from contract_heldout_ipi.eval.trace import AgentStep, EpisodeRun, ToolCallRecord

from .protocol import AgentContext, AgentPolicy, FinishAction, ToolAction


class EpisodeRunner:
    """Run an agent against the environment declared by an episode contract."""

    def run(self, episode: EpisodeContract, agent: AgentPolicy) -> EpisodeRun:
        sandbox = EmailSandbox.from_episode(episode)
        steps: list[AgentStep] = []
        max_steps = episode.budget.max_steps or episode.budget.max_tool_calls + 1

        try:
            agent.reset(episode)
        except Exception as exc:
            return self._terminated_run(
                episode,
                steps,
                "agent_error",
                agent_error=self._format_error(exc),
            )

        while len(steps) < max_steps:
            context = AgentContext(episode=episode, steps=list(steps))
            try:
                action = agent.next_action(context)
            except Exception as exc:
                return self._terminated_run(
                    episode,
                    steps,
                    "agent_error",
                    agent_error=self._format_error(exc),
                )

            step_index = len(steps)
            if isinstance(action, FinishAction):
                steps.append(AgentStep(index=step_index, content=action.answer))
                return EpisodeRun(
                    episode_id=episode.episode_id,
                    final_answer=action.answer,
                    steps=steps,
                    terminated_reason="completed",
                )

            if not isinstance(action, ToolAction):
                return self._terminated_run(
                    episode,
                    steps,
                    "agent_error",
                    agent_error=f"invalid agent action: {type(action).__name__}",
                )

            record, budget_exhausted = self._execute_tool(sandbox, action)
            steps.append(AgentStep(index=step_index, tool_call=record))
            if budget_exhausted:
                return self._terminated_run(episode, steps, "max_tool_calls")

        return self._terminated_run(episode, steps, "max_steps")

    @staticmethod
    def _execute_tool(
        sandbox: EmailSandbox, action: ToolAction
    ) -> tuple[ToolCallRecord, bool]:
        arguments = dict(action.arguments)
        try:
            result: Any = sandbox.call_tool(action.tool, arguments)
        except ToolBudgetExceeded as exc:
            return (
                ToolCallRecord(
                    tool=action.tool,
                    arguments=arguments,
                    error=f"{type(exc).__name__}: {exc}",
                ),
                True,
            )
        except Exception as exc:
            return (
                ToolCallRecord(
                    tool=action.tool,
                    arguments=arguments,
                    error=f"{type(exc).__name__}: {exc}",
                ),
                False,
            )
        return (
            ToolCallRecord(
                tool=action.tool,
                arguments=arguments,
                result=result,
            ),
            False,
        )

    @staticmethod
    def _terminated_run(
        episode: EpisodeContract,
        steps: list[AgentStep],
        reason: str,
        *,
        agent_error: str | None = None,
    ) -> EpisodeRun:
        return EpisodeRun(
            episode_id=episode.episode_id,
            final_answer="",
            steps=steps,
            terminated_reason=reason,
            agent_error=agent_error,
        )

    @staticmethod
    def _format_error(exc: Exception) -> str:
        return f"{type(exc).__name__}: {exc}"
