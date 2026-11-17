"""Experimental grouped PROSE generation and all-population approval evaluation."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import re
import statistics
from typing import Any, Callable, Mapping, Sequence

from PROSE.datasets.abortion_adapter import (
    DEFAULT_DATASET_PATH,
    DEFAULT_SOURCE_PATH,
    load_abortion_agents,
)
from PROSE.queries.approval_scorers import (
    OllamaApprovalScorer,
    OllamaApprovalScoringError,
)
from PROSE.queries.fast_queries import FastAgent, FastGenerator, GenerationFormatError
from PROSE.utils.ollama_provider import OllamaProvider, OllamaProviderError
from calibration.participant_group_selection import load_validated_embeddings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARIES = PROJECT_ROOT / "calibration" / "participant_summaries_all_100.json"
DEFAULT_EMBEDDINGS = (
    PROJECT_ROOT / "calibration" / "participant_embeddings_qwen3_0.6b.json"
)
DEFAULT_BALANCED_GROUPS = (
    PROJECT_ROOT / "calibration" / "balanced_kmeans_groups.json"
)
DEFAULT_NEAREST_GROUPS = (
    PROJECT_ROOT / "calibration" / "nearest_neighbor_groups.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "calibration" / "grouped_prose_experiment"
DEFAULT_MODEL = "qwen3:8b"


class ExperimentalRunnerError(ValueError):
    """Raised when experimental runner inputs or results are invalid."""


class _PromptAgent:
    """Expose a summary with a non-identifying label to FastGenerator."""

    def __init__(self, label: str, description: str) -> None:
        self._label = label
        self._description = description

    def get_id(self) -> str:
        return self._label

    def get_description(self) -> str:
        return self._description


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentalRunnerError(f"Could not read JSON input {path}: {exc}") from exc


def load_summaries(path: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json(path)
    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list) or not records:
        raise ExperimentalRunnerError("Summary input must contain a non-empty records list")
    summaries: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ExperimentalRunnerError("Summary records must be objects")
        participant_id = record.get("participant_id")
        summary = record.get("summary")
        if (
            not isinstance(participant_id, str)
            or not participant_id
            or participant_id in summaries
            or not isinstance(summary, dict)
        ):
            raise ExperimentalRunnerError("Summary records must have unique IDs and objects")
        summaries[participant_id] = summary
    if len(summaries) != 100:
        raise ExperimentalRunnerError(f"Expected 100 summaries, found {len(summaries)}")
    return summaries


def _group_records(payload: Any, source: str) -> list[dict[str, Any]]:
    if source == "balanced":
        groups = payload.get("groups") if isinstance(payload, dict) else None
        if not isinstance(groups, list):
            raise ExperimentalRunnerError("Balanced group input must contain groups")
        return groups
    experiments = payload.get("experiments") if isinstance(payload, dict) else None
    if not isinstance(experiments, list):
        raise ExperimentalRunnerError("Nearest group input must contain experiments")
    records: list[dict[str, Any]] = []
    for experiment in experiments:
        if not isinstance(experiment, dict):
            raise ExperimentalRunnerError("Nearest experiments must be objects")
        group_size = experiment.get("group_size")
        for group in experiment.get("groups", []):
            if not isinstance(group, dict):
                raise ExperimentalRunnerError("Nearest groups must be objects")
            copied = dict(group)
            copied["experiment_group_size"] = group_size
            records.append(copied)
    return records


def load_generation_groups(
    balanced_path: Path, nearest_path: Path, participant_ids: set[str]
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for source, path in (("balanced", balanced_path), ("nearest", nearest_path)):
        for group in _group_records(_read_json(path), source):
            ids = group.get("participant_ids")
            if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)):
                raise ExperimentalRunnerError(f"Invalid participant_ids in {source} group")
            if not set(ids).issubset(participant_ids):
                raise ExperimentalRunnerError(f"{source} group contains unknown participant IDs")
            groups.append(
                {
                    "group_id": (
                        f"{source}_{group.get('experiment_group_size', '')}_"
                        f"{group.get('group_index', len(groups))}"
                    ).rstrip("_"),
                    "approach": source,
                    "participant_ids": [str(participant_id) for participant_id in ids],
                }
            )
    return groups


def summarize_scores(scores: Mapping[str, float]) -> dict[str, Any]:
    values = list(scores.values())
    if not values:
        return {
            "successful_count": 0,
            "mean": None,
            "median": None,
            "minimum": None,
            "distribution": {},
        }
    return {
        "successful_count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "distribution": {
            str(score): count for score, count in sorted(Counter(values).items())
        },
    }


def _tokens(statement: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", statement.lower()))


def statement_similarity(statements: Sequence[str]) -> dict[str, Any]:
    pairs: list[float] = []
    for index, first in enumerate(statements):
        first_tokens = _tokens(first)
        for second in statements[index + 1 :]:
            second_tokens = _tokens(second)
            union = first_tokens | second_tokens
            pairs.append(len(first_tokens & second_tokens) / len(union) if union else 1.0)
    return {
        "statement_count": len(statements),
        "unique_statement_count": len(set(statements)),
        "duplicate_count": len(statements) - len(set(statements)),
        "mean_pairwise_token_jaccard": statistics.mean(pairs) if pairs else None,
    }


def _validate_score(score: Any) -> float:
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ExperimentalRunnerError("Approval scorer returned a non-numeric score")
    score = float(score)
    if not math.isfinite(score) or not 1 <= score <= 6:
        raise ExperimentalRunnerError("Approval scorer returned a score outside 1-6")
    return score


def _evaluate_statement(
    statement: str,
    all_agents: Mapping[str, FastAgent],
    group_ids: set[str],
) -> dict[str, Any]:
    population_scores: dict[str, float] = {}
    group_scores: dict[str, float] = {}
    failures: list[dict[str, str]] = []
    for participant_id, agent in all_agents.items():
        try:
            score, _ = agent.get_approval(statement)
            validated_score = _validate_score(score)
        except (
            OllamaApprovalScoringError,
            OllamaProviderError,
            ExperimentalRunnerError,
            ValueError,
            RuntimeError,
        ) as exc:
            failures.append(
                {"participant_id": participant_id, "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        population_scores[participant_id] = validated_score
        if participant_id in group_ids:
            group_scores[participant_id] = validated_score
    return {
        "population_scores": population_scores,
        "group_scores": group_scores,
        "population_aggregate": summarize_scores(population_scores),
        "group_aggregate": summarize_scores(group_scores),
        "scoring_success": not failures,
        "scoring_failures": failures,
    }


def _generate_candidate(
    generator: FastGenerator,
    summaries: Mapping[str, dict[str, Any]],
    participant_ids: Sequence[str],
    *,
    target_budget: int,
) -> tuple[str, int]:
    prompt_agents = [
        _PromptAgent(
            f"participant_{index}",
            json.dumps(summaries[participant_id], sort_keys=True, ensure_ascii=True),
        )
        for index, participant_id in enumerate(participant_ids, start=1)
    ]
    statements, _ = generator.generate(
        prompt_agents, target_approval=None, target_budget=target_budget
    )
    if len(statements) != 1 or not isinstance(statements[0], str) or not statements[0].strip():
        raise ExperimentalRunnerError("Generator did not return exactly one statement")
    return statements[0].strip(), 1


def _generate_candidate_with_retries(
    generator: FastGenerator,
    summaries: Mapping[str, dict[str, Any]],
    participant_ids: Sequence[str],
    *,
    target_budget: int,
    max_attempts: int,
) -> tuple[str, int]:
    if max_attempts < 1:
        raise ExperimentalRunnerError("max_attempts must be positive")
    last_error: GenerationFormatError | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            statement, _ = _generate_candidate(
                generator,
                summaries,
                participant_ids,
                target_budget=target_budget,
            )
            return statement, attempt
        except GenerationFormatError as exc:
            last_error = exc
    assert last_error is not None
    raise GenerationFormatError(
        f"Generator format validation failed after {max_attempts} attempts: "
        f"{last_error}"
    ) from last_error


def run_experiment(
    *,
    summaries_path: Path = DEFAULT_SUMMARIES,
    embeddings_path: Path = DEFAULT_EMBEDDINGS,
    balanced_groups_path: Path = DEFAULT_BALANCED_GROUPS,
    nearest_groups_path: Path = DEFAULT_NEAREST_GROUPS,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    source_path: Path = DEFAULT_SOURCE_PATH,
    endpoint: str = "http://127.0.0.1:11434",
    model: str = DEFAULT_MODEL,
    candidates_per_group: int = 1,
    target_budget: int = 100,
    generation_attempts: int = 3,
    seed: int = 0,
    timeout: float = 120,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    provider_factory: Callable[..., OllamaProvider] = OllamaProvider,
) -> dict[str, Any]:
    if candidates_per_group <= 0 or target_budget <= 0 or generation_attempts <= 0:
        raise ExperimentalRunnerError(
            "candidates_per_group, target_budget, and generation_attempts must be positive"
        )
    summaries = load_summaries(summaries_path)
    validated_embeddings = load_validated_embeddings(embeddings_path)
    if set(validated_embeddings.participant_ids) != set(summaries):
        raise ExperimentalRunnerError("Embedding and summary participant IDs do not match")
    generation_groups = load_generation_groups(
        balanced_groups_path, nearest_groups_path, set(summaries)
    )
    all_agents: dict[str, FastAgent] = {}
    provider = provider_factory(
        endpoint=endpoint,
        model=model,
        timeout=timeout,
        think=False,
        log_prompts=False,
        log_responses=False,
    )
    scorer = OllamaApprovalScorer(provider)

    def agent_factory(**kwargs: Any) -> FastAgent:
        kwargs.pop("specificity_coeff", None)
        return FastAgent(
            **kwargs,
            approval_scorer=scorer,
            specificity_coeff=0,
        )

    loaded_agents = load_abortion_agents(
        dataset_path=dataset_path,
        source_path=source_path,
        model=model,
        specificity_coeff=0,
        agent_factory=agent_factory,
    )
    for agent in loaded_agents:
        if agent.get_id() not in summaries or agent.get_id() in all_agents:
            raise ExperimentalRunnerError("Dataset agent IDs do not match summaries")
        agent.summary = json.dumps(
            summaries[agent.get_id()], sort_keys=True, ensure_ascii=True
        )
        all_agents[agent.get_id()] = agent

    random.seed(seed)
    generator = FastGenerator(
        text_provider=provider,
        prompt_type="v3_very_high_detail",
        temperature=0,
    )
    groups = [
        {
            "group_id": "baseline_population",
            "approach": "ungrouped_baseline",
            "participant_ids": list(summaries),
        },
        *generation_groups,
    ]
    records: list[dict[str, Any]] = []
    for group in groups:
        group_ids = group["participant_ids"]
        for candidate_index in range(candidates_per_group):
            base = {
                "candidate_id": (
                    f"{group['group_id']}_candidate_{candidate_index + 1}"
                ),
                "approach": group["approach"],
                "group_id": group["group_id"],
                "group_size": len(group_ids),
                "candidate_index": candidate_index,
            }
            try:
                statement, attempts = _generate_candidate_with_retries(
                    generator,
                    summaries,
                    group_ids,
                    target_budget=target_budget,
                    max_attempts=generation_attempts,
                )
            except (
                GenerationFormatError,
                OllamaProviderError,
                ExperimentalRunnerError,
                ValueError,
                RuntimeError,
            ) as exc:
                records.append(
                    {
                        **base,
                        "generation": {
                            "success": False,
                            "error": f"{type(exc).__name__}: {exc}",
                            "attempts": generation_attempts,
                        },
                        "approval_scoring": {
                            "success": False,
                            "error": "not_run_generation_failed",
                        },
                    }
                )
                continue
            evaluation = _evaluate_statement(statement, all_agents, set(group_ids))
            records.append(
                {
                    **base,
                    "generation": {
                        "success": True,
                        "statement": statement,
                        "attempts": attempts,
                    },
                    "approval_scoring": {
                        "success": evaluation["scoring_success"],
                        "failure_count": len(evaluation["scoring_failures"]),
                    },
                    "evaluation": evaluation,
                }
            )

    successful_statements = [
        record["generation"]["statement"]
        for record in records
        if record["generation"]["success"]
    ]
    output = {
        "schema_version": "experimental-grouped-prose-v1",
        "model": model,
        "provider": "ollama",
        "seed": seed,
        "candidates_per_group": candidates_per_group,
        "target_budget": target_budget,
        "generation_attempts": generation_attempts,
        "participant_count": len(all_agents),
        "records": records,
        "statement_diversity": statement_similarity(successful_statements),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    _write_report(output, output_dir / "report.md")
    return output


def _write_report(output: Mapping[str, Any], path: Path) -> None:
    approach_rows: dict[str, list[dict[str, Any]]] = {}
    for record in output["records"]:
        approach_rows.setdefault(record["approach"], []).append(record)
    lines = [
        "# Experimental Grouped PROSE Generation Report",
        "",
        "This is an experimental integration and does not modify the production PROSE workflow.",
        "",
        f"- Model/provider: `{output['model']}` via `{output['provider']}`",
        f"- Seed: `{output['seed']}`",
        f"- Participants scored: `{output['participant_count']}` per successful candidate",
        "",
        "## Results",
        "",
        "| Approach | Candidates | Generation failures | Scoring failures | Population mean | Population median | Population minimum | Population distribution | Group mean | Group median | Group minimum |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for approach, records in sorted(approach_rows.items()):
        generation_failures = sum(
            1 for record in output["records"]
            if record["approach"] == approach and not record["generation"]["success"]
        )
        successful_records = [
            record for record in records if record["generation"]["success"]
        ]
        population_values = [
            record["evaluation"]["population_aggregate"]["mean"]
            for record in successful_records
            if record["evaluation"]["population_aggregate"]["mean"] is not None
        ]
        group_values = [
            record["evaluation"]["group_aggregate"]["mean"]
            for record in successful_records
            if record["evaluation"]["group_aggregate"]["mean"] is not None
        ]
        population_medians = [
            record["evaluation"]["population_aggregate"]["median"]
            for record in successful_records
            if record["evaluation"]["population_aggregate"]["median"] is not None
        ]
        population_minima = [
            record["evaluation"]["population_aggregate"]["minimum"]
            for record in successful_records
            if record["evaluation"]["population_aggregate"]["minimum"] is not None
        ]
        group_medians = [
            record["evaluation"]["group_aggregate"]["median"]
            for record in successful_records
            if record["evaluation"]["group_aggregate"]["median"] is not None
        ]
        group_minima = [
            record["evaluation"]["group_aggregate"]["minimum"]
            for record in successful_records
            if record["evaluation"]["group_aggregate"]["minimum"] is not None
        ]
        scoring_failures = sum(
            len(record["evaluation"]["scoring_failures"])
            for record in successful_records
        )
        population_distribution = Counter(
            score
            for record in successful_records
            for score in record["evaluation"]["population_scores"].values()
        )
        lines.append(
            f"| {approach} | {len(records)} | {generation_failures} | "
            f"{scoring_failures} | "
            f"{statistics.mean(population_values) if population_values else 'n/a'} | "
            f"{statistics.mean(population_medians) if population_medians else 'n/a'} | "
            f"{min(population_minima) if population_minima else 'n/a'} | "
            f"{dict(sorted(population_distribution.items())) if population_distribution else 'n/a'} | "
            f"{statistics.mean(group_values) if group_values else 'n/a'} | "
            f"{statistics.mean(group_medians) if group_medians else 'n/a'} | "
            f"{min(group_minima) if group_minima else 'n/a'} |"
        )
    lines.extend(
        [
            "",
            "Population approval is computed across all 100 participants. Group approval is",
            "computed only for the participants whose summaries generated that candidate.",
            "Scores remain on the existing Qwen Ollama 1-6 scale; no calibration remapping",
            "is applied here.",
            "",
            "Statement diversity uses exact-duplicate counts and pairwise token Jaccard",
            "similarity. These are descriptive measures, not policy-agreement or PROSE",
            "representation guarantees. No BJR/cBJR or approximation claim is made.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summaries", type=Path, default=DEFAULT_SUMMARIES)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--balanced-groups", type=Path, default=DEFAULT_BALANCED_GROUPS)
    parser.add_argument("--nearest-groups", type=Path, default=DEFAULT_NEAREST_GROUPS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--candidates-per-group", type=int, default=1)
    parser.add_argument("--target-budget", type=int, default=100)
    parser.add_argument("--generation-attempts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run_experiment(
        summaries_path=args.summaries,
        embeddings_path=args.embeddings,
        balanced_groups_path=args.balanced_groups,
        nearest_groups_path=args.nearest_groups,
        dataset_path=args.dataset,
        source_path=args.source,
        endpoint=args.endpoint,
        model=args.model,
        candidates_per_group=args.candidates_per_group,
        target_budget=args.target_budget,
        generation_attempts=args.generation_attempts,
        seed=args.seed,
        timeout=args.timeout,
        output_dir=args.output_dir,
    )
    print(f"Saved {len(result['records'])} candidate records to {args.output_dir}")


if __name__ == "__main__":
    main()
