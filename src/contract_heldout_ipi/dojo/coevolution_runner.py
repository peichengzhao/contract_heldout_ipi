"""Train-time bidirectional co-evolution for AgentDojo attacks and defenses."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from typing import Any, Callable, Sequence

from contract_heldout_ipi.defenses.harness import AttackHarness, DefenseHarness

from .attack_optimizer import (
    DojoAttackOptimizer,
    should_update_attack,
    should_update_attack_on_qual_failure,
    select_round_attack_optimizer_pair,
    _serialize_qualification_attempts,
)
from .attack_rules import DEFAULT_DOJO_ATTACK_RULES, DojoAttackRules
from .coevolution_protocol import (
    LEGACY_COEVOLUTION_PROTOCOL,
    CoevolutionProtocol,
    resolve_qualification_success_min,
)
from .dynamic_attack import DOJO_ATTACK_SYSTEM_PROMPT, DojoAttackPlan
from .game_rules import DojoLockedRuleSet, DojoNegotiationTranscript
from .negotiation import DojoGameNegotiator
from .payload_optimizer import DojoPayloadOptimizer
from .payload_template_library import PayloadTemplateLibrary, default_payload_template_library
from .rules_runtime import dojo_attack_rules_from_game, protocol_overlay_from_game
from .template_optimizer import DojoTemplateOptimizer
from .failed_approaches import FailedApproachMemory
from .injection import build_injection_map_from_plan
from .attack_diagnosis import harness_patch_delta
from .dynamic_runner import (
    DojoDynamicReport,
    DojoDynamicPairResult,
    DojoMetrics,
    DojoNativeResult,
    DojoQualificationRunner,
    DojoReplaySummary,
    _admitted_metrics,
    _execute_native,
    _execute_replays,
    _native_metrics,
    _representative_result,
    _success_count,
    _summarize_replays,
)
from .optimizer import DojoHarnessOptimizer, should_update_harness


@dataclass(frozen=True)
class DojoTemplateUpdateRecord:
    round_index: int
    from_version: int
    to_version: int
    rationale: str
    trigger: str


@dataclass(frozen=True)
class DojoHarnessUpdateRecord:
    user_task_id: str
    injection_task_id: str
    from_version: int
    to_version: int
    rationale: str
    trigger: str
    round_index: int


@dataclass(frozen=True)
class DojoAttackUpdateRecord:
    user_task_id: str
    injection_task_id: str
    from_version: int
    to_version: int
    rationale: str
    trigger: str
    round_index: int


@dataclass(frozen=True)
class DojoArchivedAttack:
    user_task_id: str
    injection_task_id: str
    strategy: str
    payload: str
    target_vectors: tuple[str, ...]
    admitted_injections: dict[str, str]
    round_index: int
    attack_harness_version: int
    vector_payloads: tuple[tuple[str, str], ...] = ()
    payload_formats: dict[str, str] | None = None
    source: str = "generated"
    admission_tier: str = "qualified"
    qual_successes: int | None = None


@dataclass(frozen=True)
class DojoLeagueReplayResult:
    user_task_id: str
    injection_task_id: str
    source_round_index: int
    harness_version: int
    defense_result: DojoNativeResult
    defense_replay_summary: DojoReplaySummary


@dataclass(frozen=True)
class DojoOptimizationSnapshot:
    """Per-round attack/defense state and outcomes for trajectory analysis."""

    round_index: int
    defense_harness_version: int
    attack_harness_version: int
    qualification_pairs: int
    qualified_pairs: int
    qualification_rate: float | None
    round_admitted_attack_success_rate: float | None
    round_admitted_n: int
    league_replay_attack_success_rate: float | None
    league_replay_n: int
    archive_size: int
    defense_updates_in_round: int
    attack_updates_in_round: int


@dataclass(frozen=True)
class DojoRoundSummary:
    round_index: int
    qualification_pairs: int
    qualified_pairs: int
    qualification_rate: float | None
    harness_version: int
    archive_size: int


@dataclass(frozen=True)
class DojoPayloadRefineRecord:
    user_task_id: str
    injection_task_id: str
    round_index: int
    diagnosis_label: str
    rationale: str
    trigger: str
    produced_seed: bool
    produced_seed_count: int = 0
    same_round_breakthrough: bool = False
    immediate_retest_blocked: bool | None = None


@dataclass(frozen=True)
class DojoCoevolutionReport:
    schema_version: int
    created_at: str
    optimize_enabled: bool
    train_rounds: int
    max_archive_size: int
    initial_harness_version: int
    final_harness_version: int
    final_system_prompt: str
    initial_attack_harness_version: int
    final_attack_harness_version: int
    final_attack_prompt: str
    initial_template_library_version: int
    final_template_library_version: int
    coevolution_protocol: str
    negotiate_each_round: bool
    negotiation_transcripts: tuple[DojoNegotiationTranscript, ...]
    harness_updates: tuple[DojoHarnessUpdateRecord, ...]
    attack_updates: tuple[DojoAttackUpdateRecord, ...]
    payload_refines: tuple[DojoPayloadRefineRecord, ...]
    template_updates: tuple[DojoTemplateUpdateRecord, ...]
    archive: tuple[DojoArchivedAttack, ...]
    league_replays: tuple[DojoLeagueReplayResult, ...]
    final_league_replay: DojoMetrics
    optimization_trajectory: tuple[DojoOptimizationSnapshot, ...]
    round_summaries: tuple[DojoRoundSummary, ...]
    dynamic: DojoDynamicReport

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dynamic"] = self.dynamic.to_dict()
        payload["negotiation_transcripts"] = [
            item.model_dump(mode="json") for item in self.negotiation_transcripts
        ]
        return payload


PipelineFactory = Callable[[str | None], tuple[Any, str]]


class DojoCoevolutionRunner:
    """Run multi-round dynamic attacks with defense and attack prompt co-evolution."""

    def __init__(
        self,
        *,
        suite: Any,
        generator: Any,
        qualification_pipeline: Any,
        pipeline_factory: PipelineFactory,
        initial_harness: DefenseHarness,
        initial_attack_harness: AttackHarness | None = None,
        optimizer: DojoHarnessOptimizer | None = None,
        attack_optimizer: DojoAttackOptimizer | None = None,
        qualification_candidates: int = 3,
        qualification_replay_runs: int = 3,
        qualification_success_min: int | None = None,
        defense_replay_runs: int = 5,
        injection_vector_resolver: Callable[[Any], Sequence[str]] | None = None,
        train_rounds: int = 1,
        max_archive_size: int = 20,
        attack_memory_limit: int = 10,
        expand_injectable_vectors: bool = True,
        attack_rules: DojoAttackRules | None = None,
        coevolution_protocol: CoevolutionProtocol | None = None,
        coevolution_protocol_profile: str = "legacy",
        template_optimizer: DojoTemplateOptimizer | None = None,
        negotiator: DojoGameNegotiator | None = None,
        negotiation_rounds_per_train_round: int = 1,
        payload_optimizer: DojoPayloadOptimizer | None = None,
        payload_refine_candidates: int | None = None,
        same_round_attack_iters: int | None = None,
    ):
        if train_rounds < 1:
            raise ValueError("train_rounds must be at least 1")
        if max_archive_size < 1:
            raise ValueError("max_archive_size must be at least 1")
        if attack_memory_limit < 1:
            raise ValueError("attack_memory_limit must be at least 1")
        if negotiation_rounds_per_train_round < 1:
            raise ValueError("negotiation_rounds_per_train_round must be at least 1")
        self.suite = suite
        self.generator = generator
        self.qualification_pipeline = qualification_pipeline
        self.pipeline_factory = pipeline_factory
        self.initial_harness = initial_harness
        self.initial_attack_harness = initial_attack_harness or AttackHarness(
            name="attack-generator",
            prompt=DOJO_ATTACK_SYSTEM_PROMPT,
            version=0,
        )
        self.optimizer = optimizer
        self.attack_optimizer = attack_optimizer
        self.qualification_candidates = qualification_candidates
        self.qualification_replay_runs = qualification_replay_runs
        self.qualification_success_min = qualification_success_min
        self.defense_replay_runs = defense_replay_runs
        self.injection_vector_resolver = injection_vector_resolver
        self.train_rounds = train_rounds
        self.max_archive_size = max_archive_size
        self.attack_memory_limit = attack_memory_limit
        self.expand_injectable_vectors = expand_injectable_vectors
        self.attack_rules = attack_rules or DEFAULT_DOJO_ATTACK_RULES
        self.coevolution_protocol = coevolution_protocol or LEGACY_COEVOLUTION_PROTOCOL
        self.coevolution_protocol_profile = coevolution_protocol_profile
        self.template_optimizer = template_optimizer
        self.negotiator = negotiator
        self.negotiation_rounds_per_train_round = negotiation_rounds_per_train_round
        self.payload_optimizer = payload_optimizer
        self.payload_refine_candidates = payload_refine_candidates
        self.same_round_attack_iters = same_round_attack_iters
        self.resolved_qualification_success_min = resolve_qualification_success_min(
            self.coevolution_protocol,
            qualification_replay_runs=qualification_replay_runs,
            cli_override=qualification_success_min,
        )

    def run(
        self,
        *,
        user_task_ids: Sequence[str],
        injection_task_ids: Sequence[str],
        benchmark_version: str,
        suite_name: str,
        split: str,
        attack_model: str,
        task_model: str,
        defense_name: str,
        max_pairs: int | None = None,
        run_clean: bool = True,
        optimize: bool = True,
    ) -> DojoCoevolutionReport:
        if max_pairs is not None and max_pairs < 1:
            raise ValueError("max_pairs must be positive")
        selected_users = tuple(user_task_ids)
        selected_injections = tuple(injection_task_ids)
        pairs = [
            (user_id, injection_id)
            for user_id in selected_users
            for injection_id in selected_injections
        ]
        if max_pairs is not None:
            pairs = pairs[:max_pairs]
        active_users = tuple(dict.fromkeys(user_id for user_id, _ in pairs))
        active_injections = tuple(
            dict.fromkeys(injection_id for _, injection_id in pairs)
        )

        harness = self.initial_harness
        attack_harness = self.initial_attack_harness
        if hasattr(self.generator, "set_system_prompt"):
            self.generator.set_system_prompt(attack_harness.prompt)

        defense_pipeline, defense_harness = self.pipeline_factory(harness.prompt)
        attack_memory: list[dict[str, Any]] = []
        failed_approaches = FailedApproachMemory(
            strategy_vector_ban_threshold=(
                self.coevolution_protocol.failed_approach_strategy_ban_threshold
            ),
            ban_strategy_vector=(
                self.coevolution_protocol.failed_approach_ban_strategy_vector
            ),
        )
        template_library = (
            default_payload_template_library()
            if self.coevolution_protocol.template_library_enabled
            else PayloadTemplateLibrary()
        )
        if hasattr(self.generator, "set_template_library"):
            self.generator.set_template_library(template_library)
        initial_template_library_version = template_library.version
        runner = DojoQualificationRunner(
            suite=self.suite,
            generator=self.generator,
            qualification_pipeline=self.qualification_pipeline,
            defense_pipeline=defense_pipeline,
            defense_harness=defense_harness,
            qualification_candidates=self.qualification_candidates,
            qualification_replay_runs=self.qualification_replay_runs,
            qualification_success_min=self.resolved_qualification_success_min,
            defense_replay_runs=self.defense_replay_runs,
            injection_vector_resolver=self.injection_vector_resolver,
            attack_memory=attack_memory,
            expand_injectable_vectors=self.expand_injectable_vectors,
            attack_rules=self.attack_rules,
            near_miss_archive_min=self.coevolution_protocol.near_miss_archive_min,
            failed_approaches=failed_approaches,
        )

        clean_results: list[tuple[str, DojoNativeResult]] = []
        if run_clean:
            print(
                f"[progress] clean utility: {len(active_users)} user tasks",
                flush=True,
            )
            for user_id in active_users:
                print(f"[progress] clean start user={user_id}", flush=True)
                user_task = self.suite.get_user_task_by_id(user_id)
                clean_results.append(
                    (
                        user_id,
                        _execute_native_clean(
                            self.suite, self.qualification_pipeline, user_task
                        ),
                    )
                )
                clean_ok = clean_results[-1][1].utility is True
                print(
                    f"[progress] clean done user={user_id} utility={clean_ok}",
                    flush=True,
                )

        defense_updates: list[DojoHarnessUpdateRecord] = []
        attack_updates: list[DojoAttackUpdateRecord] = []
        payload_refines: list[DojoPayloadRefineRecord] = []
        template_updates: list[DojoTemplateUpdateRecord] = []
        archive: list[DojoArchivedAttack] = []
        payload_seeds: list[DojoAttackPlan] = []
        league_replays: list[DojoLeagueReplayResult] = []
        round_summaries: list[DojoRoundSummary] = []
        optimization_trajectory: list[DojoOptimizationSnapshot] = []
        last_pair_results: tuple[DojoDynamicPairResult, ...] = ()
        last_defense_update_round: int | None = None
        total_defense_updates = 0
        cumulative_attack_successes = 0
        negotiation_transcripts: list[DojoNegotiationTranscript] = []
        prior_locked: DojoLockedRuleSet | None = None
        active_protocol = self.coevolution_protocol
        round_defense_update_budget: int | None = None
        may_update_harness = True

        # Shared seed queue consumed by qualification runner.
        runner.payload_seeds = payload_seeds

        for round_index in range(self.train_rounds):
            print(
                f"[progress] train round {round_index + 1}/{self.train_rounds} "
                f"pairs={len(pairs)} archive={len(archive)}",
                flush=True,
            )
            if optimize and self.negotiator is not None:
                print(
                    f"[progress] negotiate round {round_index} "
                    f"(subrounds={self.negotiation_rounds_per_train_round})",
                    flush=True,
                )
                pressure = {
                    "previous_league_asr": (
                        optimization_trajectory[-1].league_replay_attack_success_rate
                        if optimization_trajectory
                        else None
                    ),
                    "previous_qualified": (
                        optimization_trajectory[-1].qualified_pairs
                        if optimization_trajectory
                        else None
                    ),
                    "defense_harness_version": harness.version,
                    "attack_harness_version": attack_harness.version,
                    "archive_size": len(archive),
                    "cumulative_attack_successes": cumulative_attack_successes,
                }
                transcript = self.negotiator.negotiate(
                    train_round_index=round_index,
                    defense_harness=harness,
                    suite_name=suite_name,
                    negotiation_rounds=self.negotiation_rounds_per_train_round,
                    prior_locked=prior_locked,
                    round_pressure=pressure,
                )
                negotiation_transcripts.append(transcript)
                print(
                    f"[progress] negotiate locked round={round_index} "
                    f"ok={transcript.locked is not None}",
                    flush=True,
                )
                prior_locked = transcript.locked
                locked_rules = transcript.locked.rules
                active_protocol = protocol_overlay_from_game(
                    self.coevolution_protocol, locked_rules
                )
                self.attack_rules = dojo_attack_rules_from_game(
                    locked_rules,
                    force_attack_optimizer_each_round=(
                        active_protocol.force_attack_prompt_optimizer_each_round
                    ),
                )
                may_update_harness = locked_rules.defense.may_update_harness
                round_defense_update_budget = (
                    locked_rules.defense.max_defense_updates_this_round
                )
                self.expand_injectable_vectors = (
                    locked_rules.attack.expand_injectable_vectors
                )
                runner.expand_injectable_vectors = self.expand_injectable_vectors
                runner.attack_rules = self.attack_rules
                runner.qualification_success_min = resolve_qualification_success_min(
                    active_protocol,
                    qualification_replay_runs=self.qualification_replay_runs,
                    cli_override=self.qualification_success_min,
                )
                if hasattr(self.generator, "attack_rules"):
                    self.generator.attack_rules = self.attack_rules
                    self.generator.max_payload_chars = min(
                        self.generator.max_payload_chars,
                        self.attack_rules.max_payload_chars,
                    )
                if hasattr(self.generator, "configure_for_rules"):
                    self.generator.configure_for_rules(self.attack_rules)

            runner.set_archive(archive)
            round_pairs: list[DojoDynamicPairResult] = []
            defense_updates_in_round = 0
            attack_updates_in_round = 0
            for pair_i, (user_id, injection_id) in enumerate(pairs, start=1):
                print(
                    f"[progress] R{round_index} pair {pair_i}/{len(pairs)} "
                    f"{user_id}+{injection_id}",
                    flush=True,
                )
                pair_result = runner._run_pair(user_id, injection_id)
                admitted = pair_result.admitted_plan is not None
                defense_ok = (
                    pair_result.defense_result.attack_success
                    if pair_result.defense_result is not None
                    else None
                )
                print(
                    f"[progress] R{round_index} pair done admitted={admitted} "
                    f"attack_success={defense_ok}",
                    flush=True,
                )
                round_pairs.append(pair_result)
                _extend_attack_memory(
                    attack_memory,
                    pair_result,
                    round_index=round_index,
                    limit=self.attack_memory_limit,
                )
                defense_result = pair_result.defense_result
                admitted = pair_result.admitted_plan is not None
                if defense_result is not None and defense_result.attack_success is True:
                    cumulative_attack_successes += 1
                archive_entry = _archive_entry_from_pair(
                    pair_result, round_index=round_index, attack_harness_version=attack_harness.version
                )
                if archive_entry is not None:
                    archive.append(archive_entry)
                    archive[:] = archive[-self.max_archive_size :]

                previous_harness_prompt = harness.prompt
                immediate_retest_blocked: bool | None = None
                if (
                    optimize
                    and self.optimizer is not None
                    and may_update_harness
                    and pair_result.admitted_plan is not None
                    and should_update_harness(defense_result)
                    and defense_result is not None
                    and (
                        round_defense_update_budget is None
                        or defense_updates_in_round < round_defense_update_budget
                    )
                    and _may_update_defense(
                        active_protocol,
                        round_index=round_index,
                        last_defense_update_round=last_defense_update_round,
                        total_defense_updates=total_defense_updates,
                        cumulative_attack_successes=cumulative_attack_successes,
                    )
                ):
                    user_task = self.suite.get_user_task_by_id(user_id)
                    injection_task = self.suite.get_injection_task_by_id(injection_id)
                    update = self.optimizer.optimize(
                        harness=harness,
                        user_task_prompt=user_task.PROMPT,
                        injection_goal=injection_task.GOAL,
                        attack_plan=pair_result.admitted_plan,
                        result=defense_result,
                        user_task_id=user_id,
                        injection_task_id=injection_id,
                        update_mode=active_protocol.defense_update_mode,
                    )
                    if update.harness.version != harness.version:
                        defense_updates.append(
                            DojoHarnessUpdateRecord(
                                user_task_id=user_id,
                                injection_task_id=injection_id,
                                from_version=harness.version,
                                to_version=update.harness.version,
                                rationale=update.rationale,
                                trigger=update.trigger,
                                round_index=round_index,
                            )
                        )
                        defense_updates_in_round += 1
                        total_defense_updates += 1
                        last_defense_update_round = round_index
                        harness = update.harness
                        defense_pipeline, defense_harness = self.pipeline_factory(
                            harness.prompt
                        )
                        runner.defense_pipeline = defense_pipeline
                        runner.defense_harness = defense_harness
                        batch = _replay_archive(
                            self.suite,
                            archive,
                            defense_pipeline,
                            defense_replay_runs=self.defense_replay_runs,
                            harness_version=harness.version,
                        )
                        league_replays.extend(batch)

                        # P0: immediately retest the triggering attack on the new harness.
                        if (
                            active_protocol.immediate_retest_after_defense_update
                            and pair_result.admitted_plan is not None
                            and pair_result.admitted_injections is not None
                        ):
                            print(
                                f"[progress] P0 immediate retest after defense "
                                f"v{harness.version} on {user_id}+{injection_id}",
                                flush=True,
                            )
                            retest = _execute_native(
                                self.suite,
                                defense_pipeline,
                                user_task,
                                injection_task=injection_task,
                                injections=pair_result.admitted_injections,
                            )
                            immediate_retest_blocked = retest.attack_success is not True
                            if immediate_retest_blocked:
                                failed_approaches.remember(
                                    pair_result.admitted_plan,
                                    reason="blocked_by_new_harness",
                                    diagnosis_label="detected",
                                    round_index=round_index,
                                    user_task_id=user_id,
                                    injection_task_id=injection_id,
                                )
                            if immediate_retest_blocked and self.payload_optimizer is not None:
                                _run_payload_feedback_cycle(
                                    suite=self.suite,
                                    runner=runner,
                                    payload_optimizer=self.payload_optimizer,
                                    attack_rules=self.attack_rules,
                                    protocol=active_protocol,
                                    payload_refine_candidates=(
                                        self.payload_refine_candidates
                                        if self.payload_refine_candidates is not None
                                        else active_protocol.payload_refine_candidates
                                    ),
                                    same_round_iters=(
                                        self.same_round_attack_iters
                                        if self.same_round_attack_iters is not None
                                        else active_protocol.same_round_attack_iters
                                    ),
                                    user_id=user_id,
                                    injection_id=injection_id,
                                    round_index=round_index,
                                    attack_plan=pair_result.admitted_plan,
                                    result=retest,
                                    injectable_vectors=pair_result.injectable_vectors,
                                    previous_defense_harness=previous_harness_prompt,
                                    defense_harness=runner.defense_harness,
                                    payload_seeds=payload_seeds,
                                    payload_refines=payload_refines,
                                    attack_memory=attack_memory,
                                    attack_memory_limit=self.attack_memory_limit,
                                    failed_approaches=failed_approaches,
                                    trigger="immediate_retest_blocked",
                                    immediate_retest_blocked=True,
                                    qualification_pipeline=self.qualification_pipeline,
                                    defense_pipeline=defense_pipeline,
                                    qualification_success_min=runner.qualification_success_min,
                                    qualification_replay_runs=self.qualification_replay_runs,
                                    defense_replay_runs=max(
                                        1, active_protocol.same_round_defense_replay_runs
                                    ),
                                )

                if (
                    optimize
                    and self.payload_optimizer is not None
                    and pair_result.admitted_plan is not None
                    and should_update_attack(defense_result, admitted=admitted)
                    and defense_result is not None
                ):
                    failed_approaches.remember(
                        pair_result.admitted_plan,
                        reason="defense_blocked",
                        diagnosis_label="detected",
                        round_index=round_index,
                        user_task_id=user_id,
                        injection_task_id=injection_id,
                    )
                    _run_payload_feedback_cycle(
                        suite=self.suite,
                        runner=runner,
                        payload_optimizer=self.payload_optimizer,
                        attack_rules=self.attack_rules,
                        protocol=active_protocol,
                        payload_refine_candidates=(
                            self.payload_refine_candidates
                            if self.payload_refine_candidates is not None
                            else active_protocol.payload_refine_candidates
                        ),
                        same_round_iters=(
                            self.same_round_attack_iters
                            if self.same_round_attack_iters is not None
                            else active_protocol.same_round_attack_iters
                        ),
                        user_id=user_id,
                        injection_id=injection_id,
                        round_index=round_index,
                        attack_plan=pair_result.admitted_plan,
                        result=defense_result,
                        injectable_vectors=pair_result.injectable_vectors,
                        previous_defense_harness=previous_harness_prompt,
                        defense_harness=runner.defense_harness,
                        payload_seeds=payload_seeds,
                        payload_refines=payload_refines,
                        attack_memory=attack_memory,
                        attack_memory_limit=self.attack_memory_limit,
                        failed_approaches=failed_approaches,
                        trigger="defense_blocked",
                        immediate_retest_blocked=immediate_retest_blocked,
                        qualification_pipeline=self.qualification_pipeline,
                        defense_pipeline=runner.defense_pipeline,
                        qualification_success_min=runner.qualification_success_min,
                        qualification_replay_runs=self.qualification_replay_runs,
                        defense_replay_runs=max(
                            1, active_protocol.same_round_defense_replay_runs
                        ),
                    )

                # Attack-prompt updates: skip only when payload refine both seeded
                # and broke through in-round (seed-without-breakthrough still upgrades).
                if (
                    optimize
                    and self.attack_optimizer is not None
                    and pair_result.admitted_plan is not None
                    and should_update_attack(defense_result, admitted=admitted)
                    and defense_result is not None
                    and _should_run_attack_prompt_optimizer(
                        active_protocol, "defense_blocked"
                    )
                    and not _should_skip_attack_prompt_for_payload_seed(
                        active_protocol,
                        payload_refines,
                        user_task_id=user_id,
                        injection_task_id=injection_id,
                        round_index=round_index,
                        triggers=("defense_blocked", "immediate_retest_blocked"),
                    )
                ):
                    user_task = self.suite.get_user_task_by_id(user_id)
                    injection_task = self.suite.get_injection_task_by_id(injection_id)
                    attack_update = self.attack_optimizer.optimize(
                        harness=attack_harness,
                        user_task_prompt=user_task.PROMPT,
                        injection_goal=injection_task.GOAL,
                        attack_plan=pair_result.admitted_plan,
                        defense_harness=runner.defense_harness,
                        result=defense_result,
                        user_task_id=user_id,
                        injection_task_id=injection_id,
                        trigger="defense_blocked",
                    )
                    attack_harness, attack_updates_in_round = _apply_attack_update(
                        attack_harness,
                        attack_update,
                        attack_updates,
                        attack_updates_in_round,
                        round_index=round_index,
                        user_task_id=user_id,
                        injection_task_id=injection_id,
                        generator=self.generator,
                    )

                if (
                    optimize
                    and self.payload_optimizer is not None
                    and self.attack_rules.attack_optimizer_on_qual_failure
                    and should_update_attack_on_qual_failure(pair_result)
                ):
                    best_attempt = next(
                        (
                            attempt
                            for attempt in reversed(pair_result.qualification_attempts)
                            if attempt.plan is not None and attempt.result is not None
                        ),
                        None,
                    )
                    if best_attempt is not None and best_attempt.plan is not None:
                        failed_approaches.remember(
                            best_attempt.plan,
                            reason="qualification_failed",
                            diagnosis_label="ignored",
                            round_index=round_index,
                            user_task_id=user_id,
                            injection_task_id=injection_id,
                        )
                        _run_payload_feedback_cycle(
                            suite=self.suite,
                            runner=runner,
                            payload_optimizer=self.payload_optimizer,
                            attack_rules=self.attack_rules,
                            protocol=active_protocol,
                            payload_refine_candidates=(
                                self.payload_refine_candidates
                                if self.payload_refine_candidates is not None
                                else active_protocol.payload_refine_candidates
                            ),
                            same_round_iters=(
                                self.same_round_attack_iters
                                if self.same_round_attack_iters is not None
                                else active_protocol.same_round_attack_iters
                            ),
                            user_id=user_id,
                            injection_id=injection_id,
                            round_index=round_index,
                            attack_plan=best_attempt.plan,
                            result=best_attempt.result,
                            injectable_vectors=pair_result.injectable_vectors,
                            previous_defense_harness=previous_harness_prompt,
                            defense_harness=runner.defense_harness,
                            payload_seeds=payload_seeds,
                            payload_refines=payload_refines,
                            attack_memory=attack_memory,
                            attack_memory_limit=self.attack_memory_limit,
                            failed_approaches=failed_approaches,
                            trigger="qualification_failed",
                            immediate_retest_blocked=None,
                            qualification_pipeline=self.qualification_pipeline,
                            defense_pipeline=runner.defense_pipeline,
                            qualification_success_min=runner.qualification_success_min,
                            qualification_replay_runs=self.qualification_replay_runs,
                            defense_replay_runs=max(
                                1, active_protocol.same_round_defense_replay_runs
                            ),
                        )

                # Qual-failure attack-prompt path: same breakthrough-gated skip.
                if (
                    optimize
                    and self.attack_optimizer is not None
                    and self.attack_rules.attack_optimizer_on_qual_failure
                    and should_update_attack_on_qual_failure(pair_result)
                    and _should_run_attack_prompt_optimizer(
                        active_protocol, "qualification_failed"
                    )
                    and not _should_skip_attack_prompt_for_payload_seed(
                        active_protocol,
                        payload_refines,
                        user_task_id=user_id,
                        injection_task_id=injection_id,
                        round_index=round_index,
                        triggers=("qualification_failed", "immediate_retest_blocked"),
                    )
                ):
                    user_task = self.suite.get_user_task_by_id(user_id)
                    injection_task = self.suite.get_injection_task_by_id(injection_id)
                    best_attempt = next(
                        (
                            attempt
                            for attempt in reversed(pair_result.qualification_attempts)
                            if attempt.plan is not None and attempt.result is not None
                        ),
                        None,
                    )
                    attack_update = self.attack_optimizer.optimize(
                        harness=attack_harness,
                        user_task_prompt=user_task.PROMPT,
                        injection_goal=injection_task.GOAL,
                        attack_plan=best_attempt.plan if best_attempt else None,
                        defense_harness=runner.defense_harness,
                        result=best_attempt.result if best_attempt else None,
                        user_task_id=user_id,
                        injection_task_id=injection_id,
                        trigger="qualification_failed",
                        qualification_attempts=_serialize_qualification_attempts(
                            pair_result
                        ),
                    )
                    attack_harness, attack_updates_in_round = _apply_attack_update(
                        attack_harness,
                        attack_update,
                        attack_updates,
                        attack_updates_in_round,
                        round_index=round_index,
                        user_task_id=user_id,
                        injection_task_id=injection_id,
                        generator=self.generator,
                    )

            qualified = sum(result.qualified for result in round_pairs)
            round_admitted = _admitted_metrics(round_pairs)
            round_league_batch = (
                _replay_archive(
                    self.suite,
                    archive,
                    runner.defense_pipeline,
                    defense_replay_runs=self.defense_replay_runs,
                    harness_version=harness.version,
                )
                if archive
                else []
            )
            round_league_metrics = _league_metrics(round_league_batch)
            if (
                optimize
                and self.attack_optimizer is not None
                and (
                    self.attack_rules.force_attack_optimizer_each_round
                    or active_protocol.force_attack_prompt_optimizer_each_round
                )
                and attack_updates_in_round == 0
                and round_pairs
            ):
                context_pair = select_round_attack_optimizer_pair(round_pairs)
                if context_pair is not None:
                    user_task = self.suite.get_user_task_by_id(context_pair.user_task_id)
                    injection_task = self.suite.get_injection_task_by_id(
                        context_pair.injection_task_id
                    )
                    attack_update = self.attack_optimizer.optimize(
                        harness=attack_harness,
                        user_task_prompt=user_task.PROMPT,
                        injection_goal=injection_task.GOAL,
                        attack_plan=context_pair.admitted_plan,
                        defense_harness=runner.defense_harness,
                        result=context_pair.defense_result,
                        user_task_id=context_pair.user_task_id,
                        injection_task_id=context_pair.injection_task_id,
                        trigger="round_forced",
                        round_summary={
                            "round_index": round_index,
                            "qualification_pairs": len(round_pairs),
                            "qualified_pairs": qualified,
                            "defense_harness_version": harness.version,
                            "attack_harness_version": attack_harness.version,
                        },
                        qualification_attempts=_serialize_qualification_attempts(
                            context_pair
                        ),
                    )
                    attack_harness, attack_updates_in_round = _apply_attack_update(
                        attack_harness,
                        attack_update,
                        attack_updates,
                        attack_updates_in_round,
                        round_index=round_index,
                        user_task_id=context_pair.user_task_id,
                        injection_task_id=context_pair.injection_task_id,
                        generator=self.generator,
                    )
            if (
                optimize
                and self.template_optimizer is not None
                and active_protocol.template_optimizer_each_round
            ):
                template_update = self.template_optimizer.optimize(
                    library=template_library,
                    defense_harness=runner.defense_harness,
                    round_summary={
                        "round_index": round_index,
                        "qualification_pairs": len(round_pairs),
                        "qualified_pairs": qualified,
                        "archived_pairs": sum(
                            1
                            for item in round_pairs
                            if getattr(item, "archived", item.qualified)
                        ),
                        "defense_harness_version": harness.version,
                        "attack_harness_version": attack_harness.version,
                    },
                    round_pairs=[_pair_summary(item) for item in round_pairs],
                )
                if template_update.library.version != template_library.version:
                    template_updates.append(
                        DojoTemplateUpdateRecord(
                            round_index=round_index,
                            from_version=template_library.version,
                            to_version=template_update.library.version,
                            rationale=template_update.rationale,
                            trigger=template_update.trigger,
                        )
                    )
                    template_library = template_update.library
                    if hasattr(self.generator, "set_template_library"):
                        self.generator.set_template_library(template_library)
            optimization_trajectory.append(
                DojoOptimizationSnapshot(
                    round_index=round_index,
                    defense_harness_version=harness.version,
                    attack_harness_version=attack_harness.version,
                    qualification_pairs=len(round_pairs),
                    qualified_pairs=qualified,
                    qualification_rate=(
                        qualified / len(round_pairs) if round_pairs else None
                    ),
                    round_admitted_attack_success_rate=round_admitted.attack_success_rate,
                    round_admitted_n=round_admitted.n,
                    league_replay_attack_success_rate=round_league_metrics.attack_success_rate,
                    league_replay_n=round_league_metrics.n,
                    archive_size=len(archive),
                    defense_updates_in_round=defense_updates_in_round,
                    attack_updates_in_round=attack_updates_in_round,
                )
            )
            round_summaries.append(
                DojoRoundSummary(
                    round_index=round_index,
                    qualification_pairs=len(round_pairs),
                    qualified_pairs=qualified,
                    qualification_rate=(
                        qualified / len(round_pairs) if round_pairs else None
                    ),
                    harness_version=harness.version,
                    archive_size=len(archive),
                )
            )
            last_pair_results = tuple(round_pairs)

        if archive:
            final_batch = _replay_archive(
                self.suite,
                archive,
                runner.defense_pipeline,
                defense_replay_runs=self.defense_replay_runs,
                harness_version=harness.version,
            )
            league_replays.extend(final_batch)
            final_league_replay = _league_metrics(final_batch)
        else:
            final_league_replay = DojoMetrics(0, 0, None, None, None)

        qualified = sum(result.qualified for result in last_pair_results)
        dynamic = DojoDynamicReport(
            schema_version=2,
            created_at=datetime.now(timezone.utc).isoformat(),
            agentdojo_version=version("agentdojo"),
            benchmark_version=benchmark_version,
            suite=suite_name,
            split=split,
            attack_model=attack_model,
            task_model=task_model,
            defense=defense_name,
            defense_harness=runner.defense_harness,
            selected_user_tasks=active_users,
            selected_injection_tasks=active_injections,
            qualification_candidates=self.qualification_candidates,
            qualification_replay_runs=self.qualification_replay_runs,
            qualification_success_min=runner.qualification_success_min,
            defense_replay_runs=self.defense_replay_runs,
            qualification_rate=(qualified / len(last_pair_results) if last_pair_results else None),
            qualification_pairs=len(last_pair_results),
            qualified_pairs=qualified,
            clean=_native_metrics(
                [result for _, result in clean_results], injected=False
            ),
            admitted=_admitted_metrics(last_pair_results),
            clean_results=tuple(clean_results),
            pairs=last_pair_results,
        )
        return DojoCoevolutionReport(
            schema_version=4,
            created_at=dynamic.created_at,
            optimize_enabled=optimize
            and (self.optimizer is not None or self.attack_optimizer is not None),
            train_rounds=self.train_rounds,
            max_archive_size=self.max_archive_size,
            initial_harness_version=self.initial_harness.version,
            final_harness_version=harness.version,
            final_system_prompt=harness.prompt,
            initial_attack_harness_version=self.initial_attack_harness.version,
            final_attack_harness_version=attack_harness.version,
            final_attack_prompt=attack_harness.prompt,
            initial_template_library_version=initial_template_library_version,
            final_template_library_version=template_library.version,
            coevolution_protocol=self.coevolution_protocol_profile,
            negotiate_each_round=self.negotiator is not None,
            negotiation_transcripts=tuple(negotiation_transcripts),
            harness_updates=tuple(defense_updates),
            attack_updates=tuple(attack_updates),
            payload_refines=tuple(payload_refines),
            template_updates=tuple(template_updates),
            archive=tuple(archive),
            league_replays=tuple(league_replays),
            final_league_replay=final_league_replay,
            optimization_trajectory=tuple(optimization_trajectory),
            round_summaries=tuple(round_summaries),
            dynamic=dynamic,
        )


def _run_payload_feedback_cycle(
    *,
    suite: Any,
    runner: DojoQualificationRunner,
    payload_optimizer: DojoPayloadOptimizer,
    attack_rules: DojoAttackRules,
    protocol: CoevolutionProtocol,
    payload_refine_candidates: int,
    same_round_iters: int,
    user_id: str,
    injection_id: str,
    round_index: int,
    attack_plan: DojoAttackPlan,
    result: DojoNativeResult | None,
    injectable_vectors: Sequence[str],
    previous_defense_harness: str | None,
    defense_harness: str,
    payload_seeds: list[DojoAttackPlan],
    payload_refines: list[DojoPayloadRefineRecord],
    attack_memory: list[dict[str, Any]],
    attack_memory_limit: int,
    failed_approaches: FailedApproachMemory,
    trigger: str,
    immediate_retest_blocked: bool | None,
    qualification_pipeline: Any,
    defense_pipeline: Any,
    qualification_success_min: int,
    qualification_replay_runs: int,
    defense_replay_runs: int,
) -> None:
    """Multi-candidate refine + optional same-round short iteration; ban failures."""
    user_task = suite.get_user_task_by_id(user_id)
    injection_task = suite.get_injection_task_by_id(injection_id)
    current_plan = attack_plan
    current_result = result
    breakthrough = False
    produced_total = 0
    last_diagnosis_label = "error"
    last_rationale = ""

    for iteration in range(max(1, same_round_iters + 1)):
        forbidden = sorted(
            {
                key.split("|", 1)[0]
                for key, count in failed_approaches.strategy_vector_fail_counts.items()
                if count >= failed_approaches.strategy_vector_ban_threshold
            }
        )
        refine = payload_optimizer.refine_many(
            attack_plan=current_plan,
            result=current_result,
            user_task_prompt=user_task.PROMPT,
            injection_goal=injection_task.GOAL,
            injectable_vectors=injectable_vectors,
            defense_harness=defense_harness,
            attack_rules=attack_rules,
            prior_attempts=attack_memory,
            trigger=trigger if iteration == 0 else f"{trigger}_iter{iteration}",
            previous_defense_harness=previous_defense_harness,
            failed_approaches=failed_approaches,
            forbidden_strategies=forbidden,
            n_candidates=payload_refine_candidates,
        )
        last_diagnosis_label = refine.diagnosis.label
        last_rationale = refine.rationale
        plans = list(refine.plans) if refine.plans else (
            [refine.plan] if refine.plan is not None else []
        )
        plans = failed_approaches.filter_plans(plans)
        if not plans:
            break
        produced_total += len(plans)
        for plan in plans:
            payload_seeds.append(plan)
            attack_memory.append(
                {
                    "round_index": round_index,
                    "user_task_id": user_id,
                    "injection_task_id": injection_id,
                    "source": "payload_refine",
                    "trigger": trigger,
                    "iteration": iteration,
                    "diagnosis": refine.diagnosis.to_dict(),
                    "refined_payload_preview": plan.payload[:240],
                    "strategy": plan.strategy,
                    "defense_patch_delta": harness_patch_delta(
                        previous_defense_harness, defense_harness
                    )[:400],
                }
            )
        if len(attack_memory) > attack_memory_limit:
            del attack_memory[: -attack_memory_limit]

        if same_round_iters <= 0 or iteration >= same_round_iters:
            break

        # Same-round short iteration: try refined candidates immediately.
        next_plan = None
        next_result = None
        for plan in plans:
            if failed_approaches.is_banned(plan):
                continue
            injections = build_injection_map_from_plan(plan)
            qual_replays = _execute_replays(
                suite,
                qualification_pipeline,
                user_task,
                injection_task=injection_task,
                injections=injections,
                runs=max(1, min(2, qualification_replay_runs)),
            )
            qual_successes = _success_count(qual_replays)
            if qual_successes < 1:
                failed_approaches.remember(
                    plan,
                    reason="same_round_qual_failed",
                    diagnosis_label=refine.diagnosis.label,
                    round_index=round_index,
                    user_task_id=user_id,
                    injection_task_id=injection_id,
                    # Fingerprint-only: keep strategy classes open for exploration.
                    ban_strategy_vector=False,
                )
                continue
            defense_replays = _execute_replays(
                suite,
                defense_pipeline,
                user_task,
                injection_task=injection_task,
                injections=injections,
                runs=max(1, defense_replay_runs),
            )
            defense_summary = _summarize_replays(defense_replays)
            defense_result = _representative_result(defense_replays, defense_summary)
            if defense_result.attack_success is True:
                breakthrough = True
                print(
                    f"[progress] same-round breakthrough "
                    f"{user_id}+{injection_id} strategy={plan.strategy}",
                    flush=True,
                )
                # Keep successful plan as a high-priority seed for later pairs/rounds.
                payload_seeds.insert(0, plan)
                next_plan = None
                break
            failed_approaches.remember(
                plan,
                reason="same_round_defense_blocked",
                diagnosis_label="detected",
                round_index=round_index,
                user_task_id=user_id,
                injection_task_id=injection_id,
                ban_strategy_vector=False,
            )
            next_plan = plan
            next_result = defense_result
        if breakthrough or next_plan is None or next_result is None:
            break
        current_plan = next_plan
        current_result = next_result

    payload_refines.append(
        DojoPayloadRefineRecord(
            user_task_id=user_id,
            injection_task_id=injection_id,
            round_index=round_index,
            diagnosis_label=last_diagnosis_label,
            rationale=last_rationale,
            trigger=trigger,
            produced_seed=produced_total > 0,
            produced_seed_count=produced_total,
            same_round_breakthrough=breakthrough,
            immediate_retest_blocked=immediate_retest_blocked,
        )
    )
    # Keep unused optional knobs referenced for call-site/API stability.
    _ = (runner, protocol, qualification_success_min)


def _apply_attack_update(
    attack_harness: Any,
    attack_update: Any,
    attack_updates: list[DojoAttackUpdateRecord],
    attack_updates_in_round: int,
    *,
    round_index: int,
    user_task_id: str,
    injection_task_id: str,
    generator: Any,
) -> tuple[Any, int]:
    if attack_update.harness.version == attack_harness.version:
        return attack_harness, attack_updates_in_round
    attack_updates.append(
        DojoAttackUpdateRecord(
            user_task_id=user_task_id,
            injection_task_id=injection_task_id,
            from_version=attack_harness.version,
            to_version=attack_update.harness.version,
            rationale=attack_update.rationale,
            trigger=attack_update.trigger,
            round_index=round_index,
        )
    )
    attack_harness = attack_update.harness
    if hasattr(generator, "set_system_prompt"):
        generator.set_system_prompt(attack_harness.prompt)
    return attack_harness, attack_updates_in_round + 1


def _should_run_attack_prompt_optimizer(
    protocol: CoevolutionProtocol, trigger: str
) -> bool:
    mode = protocol.attack_prompt_optimizer_mode
    if mode == "disabled":
        return False
    if mode == "defense_blocked_only":
        return trigger == "defense_blocked"
    if mode == "defense_blocked_or_qual_failure":
        return trigger in {"defense_blocked", "qualification_failed"}
    return True


def _should_skip_attack_prompt_for_payload_seed(
    protocol: CoevolutionProtocol,
    payload_refines: list[DojoPayloadRefineRecord],
    *,
    user_task_id: str,
    injection_task_id: str,
    round_index: int,
    triggers: tuple[str, ...],
) -> bool:
    """Return True when attack-prompt optimizer should be skipped.

    Legacy behavior skipped whenever a payload seed was produced. Bidirectional
    duel (and arms_race with skip_attack_prompt_only_on_breakthrough) only skips
    when that refine also broke through the current defense in-round.
    """
    if not payload_refines:
        return False
    last = payload_refines[-1]
    if (
        last.user_task_id != user_task_id
        or last.injection_task_id != injection_task_id
        or last.round_index != round_index
        or last.trigger not in triggers
        or not last.produced_seed
    ):
        return False
    if protocol.skip_attack_prompt_only_on_breakthrough:
        return bool(last.same_round_breakthrough)
    return True


def _may_update_defense(
    protocol: CoevolutionProtocol,
    *,
    round_index: int,
    last_defense_update_round: int | None,
    total_defense_updates: int,
    cumulative_attack_successes: int = 0,
) -> bool:
    if cumulative_attack_successes < protocol.defense_update_min_successes:
        return False
    if (
        protocol.max_defense_updates is not None
        and total_defense_updates >= protocol.max_defense_updates
    ):
        return False
    if last_defense_update_round is None:
        return True
    elapsed = round_index - last_defense_update_round
    return elapsed >= protocol.defense_update_cooldown_rounds


def _archive_entry_from_pair(
    pair_result: DojoDynamicPairResult,
    *,
    round_index: int,
    attack_harness_version: int,
) -> DojoArchivedAttack | None:
    if pair_result.admitted_plan is not None and pair_result.admitted_injections is not None:
        plan = pair_result.admitted_plan
        injections = pair_result.admitted_injections
        tier = "qualified"
        qual_successes = max(
            (
                _success_count(attempt.qualification_replays)
                for attempt in pair_result.qualification_attempts
                if attempt.plan is not None
            ),
            default=0,
        )
    elif (
        getattr(pair_result, "near_miss_plan", None) is not None
        and getattr(pair_result, "near_miss_injections", None) is not None
    ):
        plan = pair_result.near_miss_plan
        injections = pair_result.near_miss_injections
        tier = "near_miss"
        qual_successes = getattr(pair_result, "near_miss_successes", 0)
    else:
        return None
    return DojoArchivedAttack(
        user_task_id=pair_result.user_task_id,
        injection_task_id=pair_result.injection_task_id,
        strategy=plan.strategy,
        payload=plan.payload,
        target_vectors=plan.target_vectors,
        admitted_injections=dict(injections),
        round_index=round_index,
        attack_harness_version=attack_harness_version,
        vector_payloads=plan.vector_payloads,
        payload_formats=dict(plan.payload_formats) if plan.payload_formats else None,
        source=plan.source,
        admission_tier=tier,
        qual_successes=qual_successes,
    )


def _pair_summary(pair_result: DojoDynamicPairResult) -> dict[str, Any]:
    return {
        "user_task_id": pair_result.user_task_id,
        "injection_task_id": pair_result.injection_task_id,
        "qualified": pair_result.qualified,
        "admission_tier": getattr(pair_result, "admission_tier", None),
        "near_miss_successes": getattr(pair_result, "near_miss_successes", 0),
        "archived": getattr(pair_result, "archived", pair_result.qualified),
        "defense_attack_success": (
            pair_result.defense_result.attack_success
            if pair_result.defense_result is not None
            else None
        ),
        "attempts": _serialize_qualification_attempts(pair_result),
    }


def _extend_attack_memory(
    attack_memory: list[dict[str, Any]],
    pair: DojoDynamicPairResult,
    *,
    round_index: int,
    limit: int,
) -> None:
    for attempt in pair.qualification_attempts:
        if attempt.plan is None or attempt.result is None:
            continue
        attack_memory.append(
            {
                "round_index": round_index,
                "user_task_id": pair.user_task_id,
                "injection_task_id": pair.injection_task_id,
                "strategy": attempt.plan.strategy,
                "payload_preview": attempt.plan.payload[:240],
                "attack_success": attempt.result.attack_success,
                "qualification_successes": _success_count(
                    attempt.qualification_replays
                ),
                "observed_trajectory": attempt.result.trace.compact(),
            }
        )
    if pair.admitted_plan is not None and pair.defense_result is not None:
        summary = pair.defense_replay_summary
        attack_memory.append(
            {
                "round_index": round_index,
                "user_task_id": pair.user_task_id,
                "injection_task_id": pair.injection_task_id,
                "strategy": pair.admitted_plan.strategy,
                "payload_preview": pair.admitted_plan.payload[:240],
                "admitted": True,
                "defense_attack_success": pair.defense_result.attack_success,
                "defense_replay_asr": (
                    summary.attack_success_rate if summary is not None else None
                ),
            }
        )
    if len(attack_memory) > limit:
        del attack_memory[:-limit]


def _replay_archive(
    suite: Any,
    archive: Sequence[DojoArchivedAttack],
    defense_pipeline: Any,
    *,
    defense_replay_runs: int,
    harness_version: int,
) -> list[DojoLeagueReplayResult]:
    results: list[DojoLeagueReplayResult] = []
    for item in archive:
        user_task = suite.get_user_task_by_id(item.user_task_id)
        injection_task = suite.get_injection_task_by_id(item.injection_task_id)
        replays = _execute_replays(
            suite,
            defense_pipeline,
            user_task,
            injection_task=injection_task,
            injections=dict(item.admitted_injections),
            runs=defense_replay_runs,
        )
        summary = _summarize_replays(replays)
        defense_result = _representative_result(replays, summary)
        results.append(
            DojoLeagueReplayResult(
                user_task_id=item.user_task_id,
                injection_task_id=item.injection_task_id,
                source_round_index=item.round_index,
                harness_version=harness_version,
                defense_result=defense_result,
                defense_replay_summary=summary,
            )
        )
    return results


def _league_metrics(replays: Sequence[DojoLeagueReplayResult]) -> DojoMetrics:
    summaries = [item.defense_replay_summary for item in replays]
    if not summaries:
        return DojoMetrics(0, 0, None, None, None)
    errors = sum(summary.errors for summary in summaries)
    utility_rates = [
        summary.utility_rate
        for summary in summaries
        if summary.utility_rate is not None
    ]
    attack_rates = [
        summary.attack_success_rate
        for summary in summaries
        if summary.attack_success_rate is not None
    ]
    utility = sum(utility_rates) / len(utility_rates) if utility_rates else None
    if not attack_rates:
        return DojoMetrics(len(summaries), errors, utility, None, None)
    from .scoring import aggregate_injection_rates

    asr, defense_rate = aggregate_injection_rates(attack_rates)
    return DojoMetrics(len(summaries), errors, utility, defense_rate, asr)


def format_optimization_trajectory(
    snapshots: Sequence[DojoOptimizationSnapshot],
) -> str:
    headers = (
        "R",
        "DefV",
        "AtkV",
        "Qual",
        "AdmASR",
        "LgASR",
        "Arc",
        "D+",
        "A+",
    )
    lines = [
        " | ".join(f"{header:>6}" for header in headers),
        "-+-".join("-" * 6 for _ in headers),
    ]
    for snap in snapshots:
        qual = (
            f"{snap.qualified_pairs}/{snap.qualification_pairs}"
            if snap.qualification_pairs
            else "N/A"
        )
        adm = _fmt_rate(snap.round_admitted_attack_success_rate)
        league = _fmt_rate(snap.league_replay_attack_success_rate)
        lines.append(
            " | ".join(
                [
                    f"{snap.round_index:>6}",
                    f"{snap.defense_harness_version:>6}",
                    f"{snap.attack_harness_version:>6}",
                    f"{qual:>6}",
                    f"{adm:>6}",
                    f"{league:>6}",
                    f"{snap.archive_size:>6}",
                    f"{snap.defense_updates_in_round:>6}",
                    f"{snap.attack_updates_in_round:>6}",
                ]
            )
        )
    return "\n".join(lines)


def _fmt_rate(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def _execute_native_clean(
    suite: Any, pipeline: Any, user_task: Any
) -> DojoNativeResult:
    return _execute_native(
        suite,
        pipeline,
        user_task,
        injection_task=None,
        injections={},
    )
