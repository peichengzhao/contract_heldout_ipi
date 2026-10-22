"""Register the frozen PISmith payload as an AgentDojo attack.

Importing this module adds ``pismith`` to AgentDojo's registry. The same
pregenerated text is inserted into every visible injection slot.
"""

from __future__ import annotations

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.attacks.attack_registry import register_attack
from agentdojo.attacks.base_attacks import FixedJailbreakAttack
from agentdojo.base_tasks import BaseInjectionTask, BaseUserTask
from agentdojo.functions_runtime import Env
from agentdojo.task_suite.task_suite import TaskSuite

from contract_heldout_ipi.eval.pismith import lookup_payload


class _PismithAttack(FixedJailbreakAttack):
    name = "pismith"

    def __init__(
        self, task_suite: TaskSuite[Env], target_pipeline: BasePipelineElement
    ) -> None:
        super().__init__("{goal}", task_suite, target_pipeline)

    def attack(
        self, user_task: BaseUserTask, injection_task: BaseInjectionTask
    ) -> dict[str, str]:
        payload = lookup_payload(
            user_task.ID,
            injection_task.ID,
            suite=self.task_suite.name,
        )
        return {
            inj: payload for inj in self.get_injection_candidates(user_task)
        }


@register_attack
class PismithAttack(_PismithAttack):
    name = "pismith"
