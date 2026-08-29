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
from .attack_optimizer import (
    DojoAttackOptimizer,
    DojoAttackUpdate,
    should_update_attack,
)
from .coevolution_runner import (
    DojoArchivedAttack,
    DojoAttackUpdateRecord,
    DojoCoevolutionReport,
    DojoCoevolutionRunner,
    DojoHarnessUpdateRecord,
    DojoLeagueReplayResult,
    DojoOptimizationSnapshot,
    DojoRoundSummary,
    format_optimization_trajectory,
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
    DojoReplaySummary,
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
    "DojoAttackOptimizer",
    "DojoAttackUpdate",
    "DojoAttackUpdateRecord",
    "DojoArchivedAttack",
    "DojoCoevolutionReport",
    "DojoCoevolutionRunner",
    "DojoHarnessOptimizer",
    "DojoHarnessUpdate",
    "DojoHarnessUpdateRecord",
    "DojoLeagueReplayResult",
    "DojoOptimizationSnapshot",
    "DojoRoundSummary",
    "format_optimization_trajectory",
    "should_update_attack",
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
    "DojoReplaySummary",
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
