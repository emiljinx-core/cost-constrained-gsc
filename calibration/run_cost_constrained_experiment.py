"""Run complete abortion Algorithm 1 experiments sequentially with checkpoints."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PROSE.algorithm1_abortion import (
    CHECKPOINT_SCHEMA_V2,
    _load_checkpoint,
    load_nearest_generation_groups,
    run_algorithm1,
)
from PROSE.datasets.abortion_adapter import load_abortion_agents
from PROSE.queries.abortion_query_contracts import AbortionDiscQuery, AbortionGenQuery
from PROSE.queries.approval_scorers import OllamaApprovalScorer
from PROSE.queries.fast_queries import FastAgent, FastGenerator
from PROSE.utils.ollama_provider import OllamaProvider


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "calibration" / "participant_summaries_all_100.json"
GROUP_PATH = ROOT / "calibration" / "nearest_neighbor_groups.json"
DEFAULT_OUTPUT_DIR = ROOT / "calibration" / "cost_constrained_experiment"
DEFAULT_MODEL = "qwen3:8b"


@dataclass
class RunStatus:
    budget: int
    status: str
    output_file: str | None = None
    error: str | None = None
    runtime_seconds: float | None = None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _load_summaries() -> dict[str, Any]:
    payload = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    records = payload.get("records")
    if (
        not isinstance(records, list)
        or payload.get("participant_count") != 100
        or payload.get("successful_count") != 100
        or len(records) != 100
    ):
        raise ValueError("Expected 100 successful participant summaries")
    summaries = {record["participant_id"]: record["summary"] for record in records}
    if len(summaries) != 100:
        raise ValueError("Summary participant IDs must be unique")
    return summaries


def _check_ollama_connectivity(
    endpoint: str,
    model: str,
    timeout: float,
) -> None:
    """Abort before loading/running the experiment if Ollama is unavailable."""
    request = Request(f"{endpoint.rstrip('/')}/api/tags", method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Ollama connectivity check failed for {endpoint}: {type(exc).__name__}: {exc}"
        ) from exc
    models = payload.get("models") if isinstance(payload, dict) else None
    model_names = {
        item.get("name")
        for item in models
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    } if isinstance(models, list) else set()
    if model not in model_names:
        raise RuntimeError(
            f"Ollama is reachable at {endpoint}, but model {model!r} is unavailable"
        )


def _all_generation_failures_are_connection_errors(
    result: dict[str, Any],
) -> bool:
    """Identify a run that made no progress because every Gen call lost Ollama."""
    records = result.get("candidate_records", [])
    if not records or any(record.get("generation_success") for record in records):
        return False
    errors = [
        str(record.get("generation_error", ""))
        for record in records
        if not record.get("generation_success")
    ]
    return bool(errors) and all(
        "OllamaProviderError" in error
        or "OllamaHTTPError" in error
        or "Could not reach Ollama" in error
        for error in errors
    )


def _summary(result: dict[str, Any], runtime_seconds: float) -> dict[str, Any]:
    records = result["candidate_records"]
    return {
        "initial_population_size": result["initial_population_size"],
        "total_word_budget": result["total_word_budget"],
        "statements_generated": sum(
            record.get("generation_success", False) for record in records
        ),
        "statements_selected": len(result["selected_statements"]),
        "participants_represented": len(result["assigned_participant_ids"]),
        "participants_remaining": len(result["remaining_participant_ids"]),
        "words_consumed": result["budget_consumed"],
        "words_remaining": result["budget_remaining"],
        "budget_utilisation_percent": (
            100 * result["budget_consumed"] / result["total_word_budget"]
        ),
        "generation_failures": sum(
            not record.get("generation_success", False) for record in records
        ),
        "scoring_failures": sum(
            len(record.get("scoring_failures", [])) for record in records
        ),
        "termination_reason": result["termination_reason"],
        "runtime_seconds": runtime_seconds,
    }


def run(
    *,
    budgets: list[int],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    endpoint: str | None = None,
    model: str = DEFAULT_MODEL,
    timeout: float = 180,
    seed: int = 0,
) -> dict[str, Any]:
    if not budgets or any(budget <= 0 for budget in budgets):
        raise ValueError("budgets must contain positive integers")
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    provider = OllamaProvider(
        endpoint=endpoint,
        model=model,
        timeout=timeout,
        think=False,
        log_prompts=False,
        log_responses=False,
    )
    endpoint = provider.config.endpoint
    _check_ollama_connectivity(endpoint, model, timeout)
    summaries = _load_summaries()
    participant_ids = list(summaries)
    groups = load_nearest_generation_groups(GROUP_PATH, participant_ids=participant_ids)
    scorer = OllamaApprovalScorer(provider, max_retries=3)

    def agent_factory(**kwargs: Any) -> FastAgent:
        kwargs.pop("specificity_coeff", None)
        return FastAgent(**kwargs, approval_scorer=scorer, specificity_coeff=0)

    agents = load_abortion_agents(
        model=model,
        specificity_coeff=0,
        agent_factory=agent_factory,
        participant_ids=participant_ids,
    )
    if [agent.get_id() for agent in agents] != participant_ids:
        raise ValueError("Loaded agent IDs do not match summary IDs")
    disc_queries = {
        agent.get_id(): AbortionDiscQuery(agent) for agent in agents
    }
    gen_query = AbortionGenQuery(
        summaries,
        generator=FastGenerator(
            text_provider=provider,
            prompt_type="v3_very_high_detail",
            temperature=0,
        ),
        max_attempts=3,
    )
    disc_endpoint = scorer.provider.config.endpoint
    gen_endpoint = gen_query.generator.llm.config.endpoint
    if disc_endpoint != gen_endpoint or disc_endpoint != endpoint:
        raise RuntimeError(
            "Disc and Gen providers do not share the resolved Ollama endpoint"
        )

    progress_counts = {"generation": 0}
    progress_context: dict[str, Any] = {}

    def report_progress(event: dict[str, Any]) -> None:
        event_type = event.get("event")
        if event_type == "generation_request":
            progress_context.clear()
            progress_context.update(event)
            progress_counts["generation"] += 1
            candidate = event["candidate_number"]
            if candidate <= 20 or candidate % 100 == 0:
                print(
                    f"GEN_REQUEST budget={progress_context.get('budget')} "
                    f"threshold={event['outer_threshold']} cost={event['cost_limit']} "
                    f"group={event['generation_group']} target={event['generation_target']} "
                    f"candidate={candidate}",
                    flush=True,
                )
        elif event_type == "generation_attempt":
            candidate = progress_context.get("candidate_number", 0)
            if candidate <= 20 or event["attempt"] > 1 or candidate % 100 == 0:
                print(
                    f"GEN_ATTEMPT candidate={candidate} attempt={event['attempt']} "
                    f"target={event['target_approval']} budget={event['word_budget']}",
                    flush=True,
                )
        elif event_type == "generation_result":
            candidate = progress_context.get("candidate_number", 0)
            if (
                candidate <= 20
                or not event.get("success")
                or candidate % 100 == 0
            ):
                print(
                    f"GEN_RESULT candidate={candidate} "
                    f"success={event.get('success')} "
                    f"words={event.get('word_count', '-') } "
                    f"error={event.get('error_type', '-')} "
                    f"reason={event.get('error_message', '-')}",
                    flush=True,
                )
                if not event.get("success"):
                    save_manifest()
        elif event_type == "generation_completed":
            # Algorithm-1 level generation failures (e.g. word-count
            # disagreement, cost-limit violations) arrive here with the full
            # underlying exception message. Never reduce these to a bare type.
            candidate = event.get("candidate_number", progress_context.get("candidate_number", 0))
            if not event.get("success"):
                print(
                    f"GEN_FAILED candidate={candidate} "
                    f"error={event.get('error_type', '-')} "
                    f"reason={event.get('error_message', '-')}",
                    flush=True,
                )
                save_manifest()
        elif event_type == "cost_level":
            print(
                f"COST_LEVEL budget={progress_context.get('budget')} "
                f"threshold={event['outer_threshold']} cost={event['cost_limit']} "
                f"remaining={event['remaining_participants']} "
                f"budget_remaining={event['budget_remaining']}",
                flush=True,
            )
        elif event_type == "disc_scoring_completed":
            candidate = event["candidate_number"]
            if candidate <= 20 or event["failure_count"] or candidate % 100 == 0:
                print(
                    f"DISC_RESULT candidate={candidate} "
                    f"participants={event['participant_count']} "
                    f"supporters={event['supporter_count']} "
                    f"failures={event['failure_count']}",
                    flush=True,
                )
        elif event_type == "statement_selected":
            print(
                f"SELECTED words={event['word_count']} assigned={event['assigned_count']} "
                f"remaining={event['remaining_participants']} "
                f"budget_remaining={event['budget_remaining']}",
                flush=True,
            )
        elif event_type == "algorithm_completed":
            print(
                f"ALGORITHM_COMPLETE reason={event['termination_reason']} "
                f"selected={event['selected_count']} "
                f"remaining={event['remaining_participants']} "
                f"budget_remaining={event['budget_remaining']}",
                flush=True,
            )

    gen_query.progress_callback = report_progress

    statuses = {
        budget: RunStatus(budget=budget, status="pending") for budget in budgets
    }
    manifest_path = output_dir / "manifest.json"

    def save_manifest() -> None:
        _write_json(
            manifest_path,
            {
                "schema_version": "abortion-cost-constrained-manifest-v1",
                "model": model,
                "endpoint": endpoint,
                "disc_endpoint": disc_endpoint,
                "gen_endpoint": gen_endpoint,
                "participant_count": len(participant_ids),
                "seed": seed,
                "budgets": [asdict(statuses[budget]) for budget in budgets],
                "checkpoint_files": {
                    str(budget): f"budget_{budget}.checkpoint.json" for budget in budgets
                },
            },
        )

    def save_report() -> None:
        rows = [
            statuses[budget]
            for budget in budgets
            if statuses[budget].status == "completed"
        ]
        lines = [
            "# Cost-Constrained Abortion GSC Experiment",
            "",
            f"Model: `{model}`  ",
            f"Participants: `{len(participant_ids)}`  ",
            f"Seed: `{seed}`",
            "",
            "| Budget | Selected | Represented | Remaining | Words used | Utilisation | Gen failures | Score failures | Termination | Runtime (s) |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
        ]
        for status in rows:
            result = completed.get(str(status.budget))
            if result is None:
                continue
            summary = result["execution"]["summary"]
            lines.append(
                f"| {status.budget} | {summary['statements_selected']} | "
                f"{summary['participants_represented']} | {summary['participants_remaining']} | "
                f"{summary['words_consumed']} | {summary['budget_utilisation_percent']:.2f}% | "
                f"{summary['generation_failures']} | {summary['scoring_failures']} | "
                f"{summary['termination_reason']} | {summary['runtime_seconds']:.1f} |"
            )
        lines.extend(
            [
                "",
                "Each budget JSON contains selected statements, assignments, "
                "candidate records, failures, and budget state. Prompts and raw "
                "participant responses are not written.",
                "",
                "These empirical results do not establish theoretical "
                "approximation, proportional-representation, BJR, or cBJR guarantees.",
            ]
        )
        (output_dir / "execution_report.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    save_manifest()
    completed: dict[str, Any] = {}
    for budget in budgets:
        output_path = output_dir / f"budget_{budget}.json"
        if output_path.exists():
            statuses[budget] = RunStatus(
                budget=budget, status="completed", output_file=output_path.name
            )
            completed[str(budget)] = json.loads(output_path.read_text(encoding="utf-8"))
            save_manifest()
            continue

        checkpoint_path = output_dir / f"budget_{budget}.checkpoint.json"
        resume_state = _load_checkpoint(checkpoint_path)
        if resume_state is not None and resume_state.get("schema_version") != CHECKPOINT_SCHEMA_V2:
            # Ignore legacy/incompatible checkpoints rather than resuming from
            # an inconsistent state; the final output files are left untouched.
            resume_state = None
        if resume_state is not None:
            print(
                f"BUDGET={budget} RESUME from candidate="
                f"{resume_state.get('candidate_number')} remaining="
                f"{len(resume_state.get('remaining_participant_ids', []))} "
                f"budget_remaining={resume_state.get('budget_remaining')}",
                flush=True,
            )

        statuses[budget] = RunStatus(budget=budget, status="running")
        save_manifest()
        started = time.monotonic()
        print(f"BUDGET={budget} START participants={len(participant_ids)}", flush=True)
        try:
            result = run_algorithm1(
                participant_ids=participant_ids,
                disc_queries=disc_queries,
                gen_query=gen_query,
                generation_groups=groups,
                total_budget=budget,
                progress_callback=lambda event, budget=budget: report_progress(
                    {**event, "budget": budget}
                ),
                checkpoint_path=checkpoint_path,
                resume_state=resume_state,
            )
            if _all_generation_failures_are_connection_errors(result):
                raise RuntimeError(
                    "all generation attempts failed due to Ollama connection errors"
                )
            runtime = time.monotonic() - started
            result["execution"] = {
                "model": model,
                "seed": seed,
                "generation_group_count": len(groups),
                "runtime_seconds": runtime,
                "summary": _summary(result, runtime),
            }
            _write_json(output_path, result)
            completed[str(budget)] = result
            statuses[budget] = RunStatus(
                budget=budget,
                status="completed",
                output_file=output_path.name,
                runtime_seconds=runtime,
            )
            print(
                f"BUDGET={budget} COMPLETE selected={len(result['selected_statements'])} "
                f"represented={len(result['assigned_participant_ids'])} "
                f"remaining={len(result['remaining_participant_ids'])} "
                f"runtime={runtime:.1f}s",
                flush=True,
            )
        except KeyboardInterrupt as exc:
            runtime = time.monotonic() - started
            statuses[budget] = RunStatus(
                budget=budget,
                status="interrupted",
                error="KeyboardInterrupt",
                runtime_seconds=runtime,
            )
            save_manifest()
            save_report()
            print(f"BUDGET={budget} INTERRUPTED", flush=True)
            raise
        except Exception as exc:
            runtime = time.monotonic() - started
            statuses[budget] = RunStatus(
                budget=budget,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                runtime_seconds=runtime,
            )
            print(f"BUDGET={budget} FAILED {type(exc).__name__}: {exc}", flush=True)
            save_manifest()
            save_report()
            raise
        save_manifest()
        _write_json(
            output_dir / "combined_results.json",
            {
                "schema_version": "abortion-cost-constrained-results-v1",
                "model": model,
                "seed": seed,
                "runs": {
                    budget_key: {
                        "summary": value["execution"]["summary"],
                        "termination_reason": value["termination_reason"],
                    }
                    for budget_key, value in completed.items()
                },
            },
        )
        save_report()

    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", nargs="+", type=int, default=[500, 1000, 1500, 2000])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(
        budgets=args.budgets,
        output_dir=args.output_dir,
        endpoint=args.endpoint,
        model=args.model,
        timeout=args.timeout,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
