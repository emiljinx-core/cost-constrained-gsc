"""Empirical Algorithm 1 adaptation for abortion-policy statements."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import random
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from PROSE.queries.abortion_query_contracts import (
    AbortionDiscQuery,
    AbortionGenQuery,
    GenQueryError,
    GeneratedStatement,
)
from PROSE.utils.helper_functions import count_words


DEFAULT_NEAREST_GROUPS = Path("calibration/nearest_neighbor_groups.json")
CHECKPOINT_SCHEMA_V2 = "abortion-algorithm1-checkpoint-v2"


def _serialize_rng_state() -> dict[str, Any]:
    """Capture the global RNG state so a resume reproduces prompt shuffles."""
    version, internal, gauss_next = random.getstate()
    return {"version": version, "internal": list(internal), "gauss_next": gauss_next}


def _restore_rng_state(state: Any) -> None:
    if not state:
        return  # older checkpoints have no RNG state
    try:
        random.setstate(
            (
                int(state["version"]),
                tuple(int(value) for value in state["internal"]),
                state["gauss_next"],
            )
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise Algorithm1Error(f"invalid checkpoint RNG state: {exc}") from exc


def _save_checkpoint(checkpoint_path: Path, state: dict[str, Any]) -> None:
    """Save incremental checkpoint to disk."""
    try:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = checkpoint_path.with_suffix(".tmp")
        temp_path.write_text(
            json.dumps(state, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(checkpoint_path)
    except (OSError, TypeError, ValueError) as exc:
        # Non-fatal: log but continue execution
        print(f"Warning: checkpoint save failed: {exc}")


def _load_checkpoint(checkpoint_path: Path) -> dict[str, Any] | None:
    """Load checkpoint if it exists, return None otherwise."""
    if not checkpoint_path.exists():
        return None
    try:
        return json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Warning: checkpoint load failed: {exc}")
        return None


class Algorithm1Error(ValueError):
    """Raised when Algorithm 1 inputs are invalid."""


class DiscLike(Protocol):
    def score(self, statement: str) -> int:
        ...


class GenLike(Protocol):
    def generate(
        self, participant_ids: Sequence[str], target_approval: float, word_budget: int
    ) -> GeneratedStatement:
        ...


@dataclass(frozen=True)
class GenerationGroup:
    group_id: str
    participant_ids: tuple[str, ...]


def load_nearest_generation_groups(
    path: Path = DEFAULT_NEAREST_GROUPS,
    *,
    participant_ids: Sequence[str] | None = None,
) -> list[GenerationGroup]:
    """Load overlapping nearest-neighbour groups without changing membership."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Algorithm1Error(f"Could not read nearest-neighbour groups: {exc}") from exc
    experiments = payload.get("experiments") if isinstance(payload, dict) else None
    if not isinstance(experiments, list):
        raise Algorithm1Error("Nearest-neighbour input must contain experiments")
    allowed = set(participant_ids) if participant_ids is not None else None
    groups: list[GenerationGroup] = []
    for experiment in experiments:
        if not isinstance(experiment, dict):
            raise Algorithm1Error("Nearest-neighbour experiments must be objects")
        group_size = experiment.get("group_size")
        for group in experiment.get("groups", []):
            ids = group.get("participant_ids") if isinstance(group, dict) else None
            index = group.get("group_index") if isinstance(group, dict) else None
            if (
                not isinstance(group_size, int)
                or not isinstance(index, int)
                or not isinstance(ids, list)
                or not ids
                or len(ids) != len(set(ids))
                or any(not isinstance(identifier, str) or not identifier for identifier in ids)
            ):
                raise Algorithm1Error("Invalid nearest-neighbour group")
            if allowed is not None and not set(ids).issubset(allowed):
                raise Algorithm1Error("Nearest-neighbour group contains an unknown participant")
            groups.append(
                GenerationGroup(
                    group_id=f"nearest_{group_size}_{index}",
                    participant_ids=tuple(ids),
                )
            )
    if not groups:
        raise Algorithm1Error("Nearest-neighbour input contains no groups")
    return groups


def required_support(statement_cost: int, population_size: int, total_budget: int) -> int:
    """Return ceil(c(alpha) * n / B), the Algorithm 1 admission count."""
    if statement_cost <= 0 or population_size <= 0 or total_budget <= 0:
        raise Algorithm1Error("statement cost, population size, and budget must be positive")
    return math.ceil(statement_cost * population_size / total_budget)


def _validate_configuration(
    participant_ids: Sequence[str],
    total_budget: int,
    utility_levels: Sequence[int],
    cost_schedule: Sequence[int],
    generation_groups: Sequence[GenerationGroup],
) -> tuple[list[str], list[int], list[int]]:
    ids = list(participant_ids)
    if not ids or len(ids) != len(set(ids)) or any(not identifier for identifier in ids):
        raise Algorithm1Error("participant_ids must be unique non-empty strings")
    if not isinstance(total_budget, int) or isinstance(total_budget, bool) or total_budget <= 0:
        raise Algorithm1Error("total_budget must be a positive integer")
    levels = list(utility_levels)
    costs = list(cost_schedule)
    if not levels or levels != sorted(set(levels), reverse=True) or levels[0] <= 0:
        raise Algorithm1Error("utility_levels must be unique positive levels in descending order")
    if not costs or costs != sorted(set(costs)) or costs[0] <= 0:
        raise Algorithm1Error("cost_schedule must be unique positive levels in ascending order")
    if any(cost > total_budget for cost in costs):
        raise Algorithm1Error("cost_schedule cannot exceed total_budget")
    population = set(ids)
    if not generation_groups:
        raise Algorithm1Error("at least one generation group is required")
    for group in generation_groups:
        if not group.group_id or not group.participant_ids:
            raise Algorithm1Error("generation groups must have IDs and members")
        if len(group.participant_ids) != len(set(group.participant_ids)):
            raise Algorithm1Error("generation groups cannot contain duplicate participants")
        if not set(group.participant_ids).issubset(population):
            raise Algorithm1Error("generation group contains an unknown participant")
    return ids, levels, costs


def run_algorithm1(
    *,
    participant_ids: Sequence[str],
    disc_queries: Mapping[str, DiscLike] | Mapping[str, AbortionDiscQuery],
    gen_query: GenLike | AbortionGenQuery,
    generation_groups: Sequence[GenerationGroup],
    total_budget: int,
    utility_levels: Sequence[int] = (6, 5, 4, 3, 2, 1),
    cost_schedule: Sequence[int] | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    checkpoint_path: Path | None = None,
    resume_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the cost-constrained Algorithm 1 control flow.

    The implementation is an empirical adaptation: LLM Disc/Gen outputs are
    estimated utilities rather than the paper's exact oracle utilities.

    When ``checkpoint_path`` is provided the full run state (candidate records,
    selected statements, assignments, budget, loop cursor and score cache) is
    persisted after every candidate and every selection. Passing a checkpoint
    back via ``resume_state`` resumes after the last completed step instead of
    restarting the run.
    """
    ids, levels, costs = _validate_configuration(
        participant_ids,
        total_budget,
        utility_levels,
        list(cost_schedule) if cost_schedule is not None else list(range(1, total_budget + 1)),
        generation_groups,
    )
    if set(disc_queries) != set(ids):
        raise Algorithm1Error("disc_queries must contain exactly the population IDs")

    population_size = len(ids)
    resume_cursor: tuple[int, int, int, int] | None = None
    if resume_state is None:
        remaining = list(ids)
        assigned: set[str] = set()
        selected: list[dict[str, Any]] = []
        candidate_records: list[dict[str, Any]] = []
        budget_remaining = total_budget
        score_cache: dict[tuple[str, str], int] = {}
        termination_reason = "thresholds_exhausted"
        candidate_number = 0
    else:
        if resume_state.get("schema_version") != CHECKPOINT_SCHEMA_V2:
            raise Algorithm1Error(
                "unsupported resume checkpoint schema: "
                f"{resume_state.get('schema_version')!r}"
            )
        try:
            raw_cursor = resume_state["cursor"]
            resume_cursor = (
                int(raw_cursor["level_index"]),
                int(raw_cursor["cost_index"]),
                int(raw_cursor["target_index"]),
                int(raw_cursor["group_index"]),
            )
            candidate_records = list(resume_state["candidate_records"])
            selected = list(resume_state["selected_statements"])
            assigned = set(resume_state["assigned_participant_ids"])
            remaining = list(resume_state["remaining_participant_ids"])
            budget_remaining = int(resume_state["budget_remaining"])
            candidate_number = int(resume_state["candidate_number"])
            termination_reason = str(
                resume_state.get("termination_reason", "thresholds_exhausted")
            )
            score_cache = {
                (str(item[0]), str(item[1])): int(item[2])
                for item in resume_state.get("score_cache", [])
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise Algorithm1Error(f"invalid resume checkpoint state: {exc}") from exc
        if (
            set(remaining) | assigned != set(ids)
            or set(remaining) & assigned
            or budget_remaining < 0
            or budget_remaining > total_budget
        ):
            raise Algorithm1Error("resume checkpoint state is inconsistent with the run")
        # Restore RNG position so FastGenerator's per-candidate shuffle (and
        # therefore the generated prompts) match an uninterrupted run.
        _restore_rng_state(resume_state.get("rng_state"))

    def report(event: Mapping[str, Any]) -> None:
        if progress_callback is not None:
            progress_callback(event)

    def checkpoint(checkpoint_type: str, cursor: tuple[int, int, int, int]) -> None:
        if checkpoint_path is None:
            return
        _save_checkpoint(
            checkpoint_path,
            {
                "schema_version": CHECKPOINT_SCHEMA_V2,
                "checkpoint_type": checkpoint_type,
                "cursor": {
                    "level_index": cursor[0],
                    "cost_index": cursor[1],
                    "target_index": cursor[2],
                    "group_index": cursor[3],
                },
                "candidate_number": candidate_number,
                "selected_statements": selected,
                "candidate_records": candidate_records,
                "assigned_participant_ids": sorted(assigned),
                "remaining_participant_ids": remaining,
                "budget_remaining": budget_remaining,
                "termination_reason": termination_reason,
                "score_cache": [
                    [key[0], key[1], value] for key, value in score_cache.items()
                ],
                "rng_state": _serialize_rng_state(),
            },
        )

    for level_index, level in enumerate(levels):
        if not remaining:
            termination_reason = "population_exhausted"
            break
        target_levels = list(range(level, levels[0] + 1))
        for cost_index, cost_limit in enumerate(costs):
            if not remaining:
                termination_reason = "population_exhausted"
                break
            # Skip work already completed before the checkpoint. This check
            # must precede the budget test below so a restored low budget does
            # not prevent reaching the resume cursor. A statement_selected
            # checkpoint means this whole (threshold, cost) round is finished,
            # including its selection, so skip it as well.
            if resume_cursor is not None:
                if (level_index, cost_index) < resume_cursor[:2]:
                    continue
                if (
                    (level_index, cost_index) == resume_cursor[:2]
                    and resume_state.get("checkpoint_type") == "statement_selected"
                ):
                    continue
            if cost_limit > budget_remaining:
                termination_reason = (
                    "budget_exhausted"
                    if budget_remaining == 0
                    else "cost_schedule_exhausted"
                )
                break

            report(
                {
                    "event": "cost_level",
                    "outer_threshold": level,
                    "cost_limit": cost_limit,
                    "remaining_participants": len(remaining),
                    "budget_remaining": budget_remaining,
                }
            )
            for target_index, target_level in enumerate(target_levels):
                for group_index, group in enumerate(generation_groups):
                    position = (level_index, cost_index, target_index, group_index)
                    if resume_cursor is not None and position <= resume_cursor:
                        continue
                    subset = [identifier for identifier in group.participant_ids if identifier in remaining]
                    if not subset:
                        continue
                    base_record = {
                        "outer_threshold": level,
                        "generation_target": target_level,
                        "cost_limit": cost_limit,
                        "generation_group": group.group_id,
                        "generation_group_participant_ids": subset,
                    }
                    candidate_number += 1
                    report(
                        {
                            "event": "generation_request",
                            "candidate_number": candidate_number,
                            "outer_threshold": level,
                            "generation_target": target_level,
                            "cost_limit": cost_limit,
                            "generation_group": group.group_id,
                        }
                    )
                    try:
                        generated = gen_query.generate(subset, target_level, cost_limit)
                        statement = generated.statement
                        actual_cost = count_words(statement)
                        if actual_cost != generated.word_count:
                            raise Algorithm1Error("Gen word count disagrees with count_words")
                        if actual_cost <= 0 or actual_cost > cost_limit:
                            raise Algorithm1Error("generated statement exceeds its cost limit")
                        report(
                            {
                                "event": "generation_completed",
                                "candidate_number": candidate_number,
                                "success": True,
                                "word_count": actual_cost,
                            }
                        )
                    except (GenQueryError, Algorithm1Error, ValueError, RuntimeError) as exc:
                        report(
                            {
                                "event": "generation_completed",
                                "candidate_number": candidate_number,
                                "success": False,
                                "error_type": type(exc).__name__,
                                "error_message": str(exc),
                            }
                        )
                        candidate_records.append(
                            {
                                **base_record,
                                "generation_success": False,
                                "generation_error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        checkpoint("candidate_evaluated", position)
                        continue

                    normalized_statement = " ".join(statement.split())
                    report(
                        {
                            "event": "disc_scoring_started",
                            "candidate_number": candidate_number,
                            "participant_count": len(remaining),
                        }
                    )
                    scores: dict[str, int] = {}
                    failures: list[dict[str, str]] = []
                    for identifier in remaining:
                        cache_key = (identifier, normalized_statement)
                        try:
                            if cache_key not in score_cache:
                                score_cache[cache_key] = int(
                                    disc_queries[identifier].score(statement)
                                )
                            scores[identifier] = score_cache[cache_key]
                        except (ValueError, RuntimeError) as exc:
                            failures.append(
                                {
                                    "participant_id": identifier,
                                    "error": f"{type(exc).__name__}: {exc}",
                                }
                            )

                    supporters = [
                        identifier for identifier, score in scores.items() if score >= level
                    ]
                    required = required_support(actual_cost, population_size, total_budget)
                    qualifies = (
                        not failures
                        and actual_cost <= budget_remaining
                        and len(supporters) >= required
                    )
                    candidate_record = {
                        **base_record,
                        "generation_success": True,
                        "statement": statement,
                        "word_count": actual_cost,
                        "approval_scores": scores,
                        "scoring_failures": failures,
                        "supporters_at_threshold": supporters,
                        "supporter_count": len(supporters),
                        "required_support": required,
                        "qualified": qualifies,
                    }
                    candidate_records.append(candidate_record)
                    report(
                        {
                            "event": "disc_scoring_completed",
                            "candidate_number": candidate_number,
                            "participant_count": len(remaining),
                            "failure_count": len(failures),
                            "supporter_count": len(supporters),
                        }
                    )
                    checkpoint("candidate_evaluated", position)

            # Rebuild the candidates for this (threshold, cost) round from the
            # persisted records so a resumed run reaches the same selection.
            evaluated_candidates = [
                record
                for record in candidate_records
                if record.get("outer_threshold") == level
                and record.get("cost_limit") == cost_limit
                and record.get("generation_success")
                and not record.get("scoring_failures")
            ]
            if not evaluated_candidates:
                continue
            # Algorithm 1 (paper lines 7-8) chooses the most-approved generated
            # statement first and only then checks whether its supporters can
            # afford it. It does not search for the best *qualifying* candidate,
            # so a non-affordable winner is skipped rather than replaced.
            best = max(
                evaluated_candidates,
                key=lambda candidate: (
                    candidate["supporter_count"],
                    sum(candidate["approval_scores"].values()),
                    -candidate["word_count"],
                    candidate["statement"],
                ),
            )
            if not best["qualified"]:
                continue

            ordered_supporters = sorted(
                best["supporters_at_threshold"],
                key=lambda identifier: (-best["approval_scores"][identifier], identifier),
            )
            assigned_ids = ordered_supporters[: best["required_support"]]
            if len(assigned_ids) != best["required_support"] or assigned & set(assigned_ids):
                raise Algorithm1Error("assignment would duplicate or underfill participants")
            assigned.update(assigned_ids)
            remaining = [identifier for identifier in remaining if identifier not in assigned_ids]
            budget_remaining -= best["word_count"]
            selection = {
                "statement": best["statement"],
                "word_count": best["word_count"],
                "outer_threshold": level,
                "generation_target": best["generation_target"],
                "generation_group": best["generation_group"],
                "required_support": best["required_support"],
                "supporter_count": best["supporter_count"],
                "assigned_participant_ids": assigned_ids,
                "remaining_participant_ids": remaining.copy(),
                "budget_consumed": total_budget - budget_remaining,
                "budget_remaining": budget_remaining,
            }
            selected.append(selection)
            report(
                {
                    "event": "statement_selected",
                    "word_count": best["word_count"],
                    "assigned_count": len(assigned_ids),
                    "remaining_participants": len(remaining),
                    "budget_remaining": budget_remaining,
                }
            )
            checkpoint(
                "statement_selected",
                (
                    level_index,
                    cost_index,
                    len(target_levels) - 1,
                    len(generation_groups) - 1,
                ),
            )
            if budget_remaining == 0:
                termination_reason = "budget_exhausted"
                break
        if termination_reason == "budget_exhausted" or not remaining:
            break

    if not remaining and termination_reason == "thresholds_exhausted":
        termination_reason = "population_exhausted"
    report(
        {
            "event": "algorithm_completed",
            "termination_reason": termination_reason,
            "selected_count": len(selected),
            "remaining_participants": len(remaining),
            "budget_remaining": budget_remaining,
        }
    )
    return {
        "schema_version": "abortion-algorithm1-v1",
        "initial_population_size": population_size,
        "total_word_budget": total_budget,
        "utility_levels": levels,
        "cost_schedule": costs,
        "selected_statements": selected,
        "candidate_records": candidate_records,
        "assigned_participant_ids": sorted(assigned),
        "remaining_participant_ids": remaining,
        "budget_consumed": total_budget - budget_remaining,
        "budget_remaining": budget_remaining,
        "termination_reason": termination_reason,
    }
