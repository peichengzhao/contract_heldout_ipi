"""Register ChatInject as native AgentDojo attacks.

Importing this module adds the attacks to AgentDojo's registry. The evaluator
does that before ``load_attack``.
"""

from __future__ import annotations

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.attacks.attack_registry import register_attack
from agentdojo.attacks.base_attacks import FixedJailbreakAttack
from agentdojo.base_tasks import BaseInjectionTask, BaseUserTask
from agentdojo.functions_runtime import Env
from agentdojo.task_suite.task_suite import TaskSuite

from contract_heldout_ipi.eval.chatinject import render_chat_inject


class _ChatInjectAttack(FixedJailbreakAttack):
    style = "single"
    template = "deepseek"

    def __init__(
        self, task_suite: TaskSuite[Env], target_pipeline: BasePipelineElement
    ) -> None:
        super().__init__("{goal}", task_suite, target_pipeline)

    def attack(
        self, user_task: BaseUserTask, injection_task: BaseInjectionTask
    ) -> dict[str, str]:
        payload = render_chat_inject(
            injection_task.GOAL, style=self.style, template=self.template
        )
        return {
            inj: payload for inj in self.get_injection_candidates(user_task)
        }


@register_attack
class ChatInjectDeepSeek(_ChatInjectAttack):
    name = "chat_inject_deepseek"
    style = "single"
    template = "deepseek"


@register_attack
class ChatInjectQwen3(_ChatInjectAttack):
    name = "chat_inject_qwen3"
    style = "single"
    template = "qwen3"


@register_attack
class ChatInjectDeepSeekMultiturn(_ChatInjectAttack):
    name = "chat_inject_deepseek_multiturn"
    style = "multiturn"
    template = "deepseek"


@register_attack
class ChatInjectQwen3Multiturn(_ChatInjectAttack):
    name = "chat_inject_qwen3_multiturn"
    style = "multiturn"
    template = "qwen3"
