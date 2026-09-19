"""AgentDojo dataset and native-evaluator integration."""

from .adapter import (
    AgentDojoDependencyError,
    DojoManifest,
    DojoTask,
    build_manifest,
    expand_injectable_vectors,
    load_suite,
    select_task_ids,
)
from .evaluator import (
    DojoCaseResult,
    DojoEvaluationReport,
    DojoMetrics,
    evaluate_fixed_defense,
)
from .coevolution_protocol import (
    ARMS_RACE_COEVOLUTION_PROTOCOL,
    DUEL_COEVOLUTION_PROTOCOL,
    LEGACY_COEVOLUTION_PROTOCOL,
    CoevolutionProtocol,
    resolve_coevolution_protocol,
    resolve_qualification_success_min,
    resolve_task_temperature,
)
from .defense_bootstrap import build_bootstrap_harness
from .payload_template_library import (
    PayloadTemplate,
    PayloadTemplateLibrary,
    default_payload_template_library,
)
from .template_optimizer import DojoTemplateOptimizer, TemplateLibraryUpdate
from .attack_rules import (
    ATTACKER_ADVANTAGE_DOJO_ATTACK_RULES,
    ATTACK_RULE_PROFILES,
    DEFAULT_DOJO_ATTACK_RULES,
    DojoAttackRules,
    LEGACY_DOJO_ATTACK_RULES,
    resolve_attack_rules,
)
from .archive_mutator import plans_from_archive
from .attack_optimizer import (
    DojoAttackOptimizer,
    DojoAttackUpdate,
    should_update_attack,
    should_update_attack_on_qual_failure,
)
from .coevolution_runner import (
    DojoArchivedAttack,
    DojoAttackUpdateRecord,
    DojoCoevolutionReport,
    DojoCoevolutionRunner,
    DojoHarnessUpdateRecord,
    DojoLeagueReplayResult,
    DojoOptimizationSnapshot,
    DojoPayloadRefineRecord,
    DojoRoundSummary,
    format_optimization_trajectory,
)
from .constitution import (
    DEFAULT_DOJO_CONSTITUTION,
    DUEL_DOJO_CONSTITUTION,
    DojoConstitution,
    DojoRuleViolation,
)
from .game_rules import (
    DojoAttackGameRules,
    DojoDefenseGameRules,
    DojoGameRules,
    DojoLockedRuleSet,
    DojoNegotiationTranscript,
    DojoNegotiationTurn,
)
from .negotiation import DojoGameNegotiator
from .payload_optimizer import DojoPayloadOptimizer, PayloadRefineResult
from .attack_diagnosis import AttackDiagnosis, diagnose_attack_result, harness_patch_delta
from .failed_approaches import FailedApproachMemory, approach_fingerprint
from .rules_runtime import dojo_attack_rules_from_game, protocol_overlay_from_game
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
    "ARMS_RACE_COEVOLUTION_PROTOCOL",
    "DUEL_COEVOLUTION_PROTOCOL",
    "LEGACY_COEVOLUTION_PROTOCOL",
    "CoevolutionProtocol",
    "resolve_coevolution_protocol",
    "resolve_qualification_success_min",
    "resolve_task_temperature",
    "build_bootstrap_harness",
    "PayloadTemplate",
    "PayloadTemplateLibrary",
    "default_payload_template_library",
    "DojoTemplateOptimizer",
    "TemplateLibraryUpdate",
    "ATTACKER_ADVANTAGE_DOJO_ATTACK_RULES",
    "AgentDojoDependencyError",
    "DEFAULT_DOJO_ATTACK_RULES",
    "DojoAttackRules",
    "LEGACY_DOJO_ATTACK_RULES",
    "resolve_attack_rules",
    "plans_from_archive",
    "DOJO_DEFENSES",
    "DOJO_ATTACK_STRATEGIES",
    "CapturingPipeline",
    "DojoAttackOptimizer",
    "DojoAttackUpdate",
    "DojoAttackUpdateRecord",
    "DojoArchivedAttack",
    "DojoCoevolutionReport",
    "DojoCoevolutionRunner",
    "DojoConstitution",
    "DEFAULT_DOJO_CONSTITUTION",
    "DUEL_DOJO_CONSTITUTION",
    "DojoRuleViolation",
    "DojoAttackGameRules",
    "DojoDefenseGameRules",
    "DojoGameRules",
    "DojoLockedRuleSet",
    "DojoNegotiationTranscript",
    "DojoNegotiationTurn",
    "DojoGameNegotiator",
    "DojoPayloadOptimizer",
    "PayloadRefineResult",
    "AttackDiagnosis",
    "diagnose_attack_result",
    "harness_patch_delta",
    "FailedApproachMemory",
    "approach_fingerprint",
    "dojo_attack_rules_from_game",
    "protocol_overlay_from_game",
    "DojoHarnessOptimizer",
    "DojoPayloadRefineRecord",
    "DojoHarnessUpdate",
    "DojoHarnessUpdateRecord",
    "DojoLeagueReplayResult",
    "DojoOptimizationSnapshot",
    "DojoRoundSummary",
    "format_optimization_trajectory",
    "should_update_attack",
    "should_update_attack_on_qual_failure",
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
    "expand_injectable_vectors",
    "load_suite",
    "select_task_ids",
    "should_update_harness",
    "trace_from_messages",
    "DojoToolCallTrace",
]
