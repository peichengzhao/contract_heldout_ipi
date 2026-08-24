"""AgentDojo dataset and native-evaluator integration."""

from .adapter import (
    AgentDojoDependencyError,
    DojoManifest,
    DojoTask,
    build_manifest,
    load_suite,
    select_task_ids,
)
from .evaluator import (
    DojoCaseResult,
    DojoEvaluationReport,
    DojoMetrics,
    evaluate_fixed_defense,
)
from .dynamic_attack import (
    DOJO_ATTACK_STRATEGIES,
    DojoAttackPlan,
    DojoDynamicAttackGenerator,
)
from .dynamic_runner import (
    DojoDynamicPairResult,
    DojoDynamicReport,
    DojoNativeResult,
    DojoQualificationAttempt,
    DojoQualificationRunner,
)
from .pipeline import DOJO_DEFENSES, build_dojo_pipeline
from .trace import (
    CapturingPipeline,
    DojoExecutionTrace,
    DojoToolCallTrace,
    trace_from_messages,
)

__all__ = [
    "AgentDojoDependencyError",
    "DOJO_DEFENSES",
    "DOJO_ATTACK_STRATEGIES",
    "CapturingPipeline",
    "DojoAttackPlan",
    "DojoCaseResult",
    "DojoEvaluationReport",
    "DojoDynamicAttackGenerator",
    "DojoDynamicPairResult",
    "DojoDynamicReport",
    "DojoExecutionTrace",
    "DojoManifest",
    "DojoMetrics",
    "DojoNativeResult",
    "DojoQualificationAttempt",
    "DojoQualificationRunner",
    "DojoTask",
    "build_dojo_pipeline",
    "build_manifest",
    "evaluate_fixed_defense",
    "load_suite",
    "select_task_ids",
    "trace_from_messages",
    "DojoToolCallTrace",
]
