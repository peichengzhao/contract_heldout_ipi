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
from .coevolution_runner import (
    DojoCoevolutionReport,
    DojoCoevolutionRunner,
    DojoHarnessUpdateRecord,
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
from .optimizer import DojoHarnessOptimizer, DojoHarnessUpdate, should_update_harness
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
    "DojoCoevolutionReport",
    "DojoCoevolutionRunner",
    "DojoHarnessOptimizer",
    "DojoHarnessUpdate",
    "DojoHarnessUpdateRecord",
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
    "should_update_harness",
    "trace_from_messages",
    "DojoToolCallTrace",
]
