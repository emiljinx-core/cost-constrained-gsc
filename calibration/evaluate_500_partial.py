"""Read-only evaluation of the partial (interrupted) 500-budget Algorithm 1 run.

Source: calibration/cost_constrained_experiment_500_final/budget_500.checkpoint.json
Outputs (new, never overwriting experiment files): calibration/evaluation_500_partial/

All metrics are computed from the checkpoint snapshot plus the run manifest.
No Ollama calls, no new candidates, no Algorithm 1 / query-contract changes.
"""

from __future__ import annotations

from collections import Counter
import json
import math
import re
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "calibration" / "cost_constrained_experiment_500_final"
CHECKPOINT = RUN_DIR / "budget_500.checkpoint.json"
MANIFEST = RUN_DIR / "manifest.json"
OUT = ROOT / "calibration" / "evaluation_500_partial"

TOTAL_BUDGET = 500
EXPECTED_PARTICIPANTS = 100

STOPWORDS = set(
    "a an the and or but if then than that this these those is are was were be been being "
    "to of in on for with as at by from into about over after before it its it's i you your "
    "we our they their he she his her them us me my do does did done have has had not no nor "
    "so such can could should would may might must will just also very more most some any all "
    "each every other another own same s t don now".split()
)


def count_words(text: str) -> int:
    """Mirror PROSE.utils.helper_functions.count_words exactly."""
    return len(text.split())


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def content_tokens(text: str) -> set[str]:
    return tokens(text) - STOPWORDS


def jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def pairwise(values):
    for i, first in enumerate(values):
        for second in values[i + 1:]:
            yield first, second


def describe(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "stdev": None,
                "min": None, "max": None}
    return {
        "count": len(values),
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "stdev": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def main() -> None:
    ckpt = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)

    selected = ckpt["selected_statements"]
    records = ckpt["candidate_records"]
    successes = [r for r in records if r.get("generation_success")]
    failures = [r for r in records if not r.get("generation_success")]
    assigned = ckpt["assigned_participant_ids"]
    remaining = ckpt["remaining_participant_ids"]
    score_cache = ckpt["score_cache"]
    budget_remaining = ckpt["budget_remaining"]

    # ------------------------------------------------------------------ #
    # 0. Extracted datasets -> CSV                                         #
    # ------------------------------------------------------------------ #
    sel_rows = []
    for i, s in enumerate(selected, 1):
        sel_rows.append({
            "selection_index": i,
            "statement": s["statement"],
            "word_count": s["word_count"],
            "outer_threshold": s["outer_threshold"],
            "generation_target": s["generation_target"],
            "generation_group": s["generation_group"],
            "required_support": s["required_support"],
            "supporter_count": s["supporter_count"],
            "assigned_count": len(s["assigned_participant_ids"]),
            "assigned_participant_ids": ";".join(s["assigned_participant_ids"]),
            "budget_consumed_cumulative": s.get("budget_consumed"),
            "budget_remaining_cumulative": s.get("budget_remaining"),
        })
    pd.DataFrame(sel_rows).to_csv(OUT / "extracted_selected_statements.csv", index=False)

    rec_rows = []
    for i, r in enumerate(records, 1):
        scores = r.get("approval_scores") or {}
        rec_rows.append({
            "candidate_number": i,
            "outer_threshold": r["outer_threshold"],
            "generation_target": r["generation_target"],
            "cost_limit": r["cost_limit"],
            "generation_group": r["generation_group"],
            "generation_group_size": len(r["generation_group_participant_ids"]),
            "generation_success": r["generation_success"],
            "statement": r.get("statement", ""),
            "word_count": r.get("word_count"),
            "scored_participants": len(scores),
            "mean_approval_score": (float(statistics.mean(scores.values())) if scores else None),
            "supporter_count": r.get("supporter_count"),
            "required_support": r.get("required_support"),
            "scoring_failure_count": len(r.get("scoring_failures", [])),
            "qualified": r.get("qualified"),
            "generation_error": r.get("generation_error", ""),
        })
    pd.DataFrame(rec_rows).to_csv(OUT / "extracted_candidate_records.csv", index=False)

    approval_rows = []
    for i, r in enumerate(successes, 1):
        for participant, score in (r.get("approval_scores") or {}).items():
            approval_rows.append({
                "candidate_index": i,
                "outer_threshold": r["outer_threshold"],
                "statement": r["statement"],
                "participant_id": participant,
                "approval_score": score,
            })
    pd.DataFrame(approval_rows).to_csv(OUT / "extracted_approval_scores.csv", index=False)

    pd.DataFrame(failures).assign(
        cost_limit=lambda d: d["cost_limit"],
    ).to_csv(OUT / "extracted_generation_failures.csv", index=False)

    pd.DataFrame(score_cache, columns=["participant_id", "statement", "approval_score"]).to_csv(
        OUT / "extracted_score_cache.csv", index=False
    )

    # ------------------------------------------------------------------ #
    # 1. Query accuracy (MODEL-BASED scores, NOT human ground truth)       #
    # ------------------------------------------------------------------ #
    all_scores = [score for r in successes for score in (r.get("approval_scores") or {}).values()]
    dist = Counter(all_scores)
    modal_shares = []
    per_candidate_means = []
    supporter_fractions = []
    for r in successes:
        scores = list((r.get("approval_scores") or {}).values())
        if scores:
            counts = Counter(scores)
            modal_shares.append(max(counts.values()) / len(scores))
            per_candidate_means.append(statistics.mean(scores))
            scored_n = len(scores)
            supporter_fractions.append((r.get("supporter_count") or 0) / scored_n)

    stmt_to_records: dict[str, list[dict]] = {}
    for r in successes:
        stmt_to_records.setdefault(r["statement"], []).append(r)
    repeated = {s: rs for s, rs in stmt_to_records.items() if len(rs) > 1}

    cache_map = {(p, s): v for p, s, v in score_cache}
    cache_conflicts = 0
    checked_pairs = 0
    for r in successes:
        for participant, score in (r.get("approval_scores") or {}).items():
            key = (participant, r["statement"])
            if key in cache_map:
                checked_pairs += 1
                if cache_map[key] != score:
                    cache_conflicts += 1

    query_acc = {
        "label": "MODEL-BASED approval scores from the qwen3:8b Disc query - not human ground truth",
        "human_ground_truth_available": False,
        "scored_pairs_total": len(all_scores),
        "score_scale": {"min_observed": min(all_scores), "max_observed": max(all_scores),
                        "expected_range": [1, 6]},
        "out_of_range_scores": sum(1 for v in all_scores if v < 1 or v > 6),
        "score_distribution_counts": {str(k): dist.get(k, 0) for k in range(1, 7)},
        "score_distribution_percent": {
            str(k): round(100 * dist.get(k, 0) / len(all_scores), 3) for k in range(1, 7)
        },
        "score_summary": describe(all_scores),
        "per_candidate_mean_score": describe(per_candidate_means),
        "per_candidate_modal_score_share": describe(modal_shares),
        "per_candidate_supporter_fraction": describe(supporter_fractions),
        "consistency": {
            "unique_statements_scored": len(stmt_to_records),
            "statements_scored_more_than_once": len(repeated),
            "cached_score_pairs_checked": checked_pairs,
            "cached_score_conflicts": cache_conflicts,
            "note": (
                "Repeated statements reuse the score_cache, so identical scores are "
                "guaranteed by construction; this is NOT an independent test-retest "
                "reliability measurement. No repeated independent scoring calls exist."
            ),
        },
        "interpretation_caveat": (
            "These figures describe the LLM scorer's output distribution and internal "
            "concentration only. No human ratings exist in this project, so no accuracy "
            "relative to ground truth, no BJR/cBJR guarantee, and no calibration against "
            "human preferences can be claimed."
        ),
    }
    (OUT / "metrics_query_accuracy.json").write_text(
        json.dumps(query_acc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    pd.DataFrame(
        [{"score": k, "count": dist.get(k, 0),
          "percent": round(100 * dist.get(k, 0) / len(all_scores), 3)} for k in range(1, 7)]
    ).to_csv(OUT / "metric_query_score_distribution.csv", index=False)
    pd.DataFrame(
        [{"candidate_index": i, "mean_score": m, "modal_share": ms, "supporter_fraction": sf}
         for i, (m, ms, sf) in enumerate(zip(per_candidate_means, modal_shares, supporter_fractions), 1)]
    ).to_csv(OUT / "metric_query_per_candidate.csv", index=False)

    # ------------------------------------------------------------------ #
    # 2. Representation                                                    #
    # ------------------------------------------------------------------ #
    per_statement_assigned = [len(s["assigned_participant_ids"]) for s in selected]
    assignment_counter = Counter(assigned)
    representation = {
        "participants_total": EXPECTED_PARTICIPANTS,
        "statements_selected": len(selected),
        "participants_assigned": len(assigned),
        "unique_assigned_participants": len(set(assigned)),
        "participants_remaining": len(remaining),
        "coverage_percent": round(100 * len(assigned) / EXPECTED_PARTICIPANTS, 3),
        "remaining_percent": round(100 * len(remaining) / EXPECTED_PARTICIPANTS, 3),
        "assigned_per_statement": describe(per_statement_assigned),
        "assigned_per_statement_distribution": {
            str(k): v for k, v in sorted(Counter(per_statement_assigned).items())
        },
        "sum_of_statement_assignment_counts": sum(per_statement_assigned),
        "participants_assigned_more_than_once": {
            str(p): c for p, c in assignment_counter.items() if c > 1
        },
        "required_support_distribution": {
            str(k): v for k, v in sorted(Counter(s["required_support"] for s in selected).items())
        },
        "supporter_count_distribution": {
            str(k): v for k, v in sorted(Counter(s["supporter_count"] for s in selected).items())
        },
        "note": (
            "Each selected statement represents only the minimum required_support agents "
            "(highest-scoring supporters), not all of its supporters at the threshold."
        ),
    }
    (OUT / "metrics_representation.json").write_text(
        json.dumps(representation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    pd.DataFrame(
        [{"selection_index": i, "statement": s["statement"][:70],
          "assigned_count": len(s["assigned_participant_ids"]),
          "required_support": s["required_support"],
          "supporter_count": s["supporter_count"]}
         for i, s in enumerate(selected, 1)]
    ).to_csv(OUT / "metric_representation_per_statement.csv", index=False)

    # ------------------------------------------------------------------ #
    # 3. Statement diversity                                               #
    # ------------------------------------------------------------------ #
    sel_statements = [s["statement"] for s in selected]
    gen_statements = [r["statement"] for r in successes]
    sel_tokens = [tokens(s) for s in sel_statements]
    sel_content = [content_tokens(s) for s in sel_statements]
    jac = [jaccard(a, b) for a, b in pairwise(sel_tokens)]
    cjac = [jaccard(a, b) for a, b in pairwise(sel_content)]
    all_tokens = [tok for t in sel_tokens for tok in t]
    lengths = [count_words(s) for s in sel_statements]
    diversity = {
        "selected_unique_statements": len(set(sel_statements)),
        "selected_statement_count": len(sel_statements),
        "selected_exact_duplicates": len(sel_statements) - len(set(sel_statements)),
        "generated_unique_statements": len(set(gen_statements)),
        "generated_statement_count": len(gen_statements),
        "generated_exact_duplicates": len(gen_statements) - len(set(gen_statements)),
        "lexical_jaccard_all_tokens": describe(jac),
        "lexical_jaccard_content_tokens": describe(cjac),
        "distinct_vocabulary_size_selected": len(set(all_tokens)),
        "total_tokens_selected": len(all_tokens),
        "type_token_ratio_selected": round(len(set(all_tokens)) / len(all_tokens), 4) if all_tokens else None,
        "word_count_per_statement": describe(lengths),
        "semantic_similarity": {
            "computed": False,
            "reason": (
                "No statement embeddings exist in the project. The only available embeddings "
                "are participant embeddings (qwen3:0.6b, for participant summaries), which are "
                "not comparable to statements. Computing statement embeddings would require "
                "Ollama calls, which are explicitly out of scope for this evaluation."
            ),
        },
    }
    (OUT / "metrics_diversity.json").write_text(
        json.dumps(diversity, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    pd.DataFrame({"pairwise_jaccard": jac}).to_csv(OUT / "metric_diversity_pairwise_jaccard.csv", index=False)
    pd.DataFrame({"pairwise_content_jaccard": cjac}).to_csv(
        OUT / "metric_diversity_pairwise_content_jaccard.csv", index=False
    )

    # ------------------------------------------------------------------ #
    # 4. Budget utilisation                                                #
    # ------------------------------------------------------------------ #
    selected_word_sum = sum(s["word_count"] for s in selected)
    last_cumulative_consumed = selected[-1].get("budget_consumed") if selected else None
    last_cumulative_remaining = selected[-1].get("budget_remaining") if selected else None
    budget = {
        "total_budget_words": TOTAL_BUDGET,
        "checkpoint_budget_remaining": budget_remaining,
        "checkpoint_implied_consumed": TOTAL_BUDGET - budget_remaining,
        "sum_selected_statement_word_counts": selected_word_sum,
        "last_selection_cumulative_consumed": last_cumulative_consumed,
        "last_selection_cumulative_remaining": last_cumulative_remaining,
        "utilisation_percent": round(100 * selected_word_sum / TOTAL_BUDGET, 3),
        "words_remaining_percent": round(100 * budget_remaining / TOTAL_BUDGET, 3),
        "reconciliation": {
            "consumed_matches_checkpoint_implied": selected_word_sum == TOTAL_BUDGET - budget_remaining,
            "selected_word_counts_match_count_words": all(
                s["word_count"] == count_words(s["statement"]) for s in selected
            ),
            "cumulative_field_matches_sum": last_cumulative_consumed == selected_word_sum,
        },
        "word_count_per_statement": describe(lengths),
    }
    (OUT / "metrics_budget.json").write_text(
        json.dumps(budget, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    pd.DataFrame(
        [{"selection_index": i, "statement": s["statement"][:70], "word_count": s["word_count"],
          "cumulative_consumed": s.get("budget_consumed")}
         for i, s in enumerate(selected, 1)]
    ).to_csv(OUT / "metric_budget_cumulative.csv", index=False)

    # ------------------------------------------------------------------ #
    # 5. Runtime (only what is actually recorded)                          #
    # ------------------------------------------------------------------ #
    total_runtime = next(
        (b.get("runtime_seconds") for b in manifest["budgets"] if b.get("budget") == TOTAL_BUDGET),
        None,
    )
    runtime = {
        "source": "manifest.json (wall-clock runtime until KeyboardInterrupt)",
        "manifest_status": manifest["budgets"][0].get("status"),
        "manifest_error": manifest["budgets"][0].get("error"),
        "total_runtime_seconds": total_runtime,
        "total_runtime_minutes": round(total_runtime / 60, 3) if total_runtime else None,
        "per_event_timestamps_available": False,
        "missing_timing_data": (
            "The checkpoint contains no per-generation or per-scoring timestamps; "
            "separate average generation/scoring times CANNOT be calculated."
        ),
        "derived_aggregates_from_total_only": {
            "seconds_per_candidate_record": (
                round(total_runtime / len(records), 3) if total_runtime else None
            ),
            "seconds_per_successful_generation": (
                round(total_runtime / len(successes), 3) if total_runtime and successes else None
            ),
            "seconds_per_selected_statement": (
                round(total_runtime / len(selected), 3) if total_runtime and selected else None
            ),
            "caveat": (
                "Derived by dividing total wall-clock time by counts; includes retries, "
                "scoring, connectivity checks and interrupt handling. NOT isolated "
                "generation or scoring times."
            ),
        },
    }
    (OUT / "metrics_runtime.json").write_text(
        json.dumps(runtime, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    pd.DataFrame(
        [{"metric": k, "value_seconds": v} for k, v in {
            "total_runtime_until_interrupt": total_runtime,
            "derived_seconds_per_candidate_record": runtime["derived_aggregates_from_total_only"]["seconds_per_candidate_record"],
            "derived_seconds_per_successful_generation": runtime["derived_aggregates_from_total_only"]["seconds_per_successful_generation"],
            "derived_seconds_per_selected_statement": runtime["derived_aggregates_from_total_only"]["seconds_per_selected_statement"],
        }.items()]
    ).to_csv(OUT / "metric_runtime.csv", index=False)

    # ------------------------------------------------------------------ #
    # Validation                                                          #
    # ------------------------------------------------------------------ #
    expected_ids = {f"generation{i}" for i in range(1, EXPECTED_PARTICIPANTS + 1)}
    assigned_set, remaining_set = set(assigned), set(remaining)
    checks = [
        ("no_duplicate_assignments", len(assigned) == len(assigned_set)),
        ("no_duplicate_within_statement_assignments",
         all(len(s["assigned_participant_ids"]) == len(set(s["assigned_participant_ids"]))
             for s in selected)),
        ("assigned_plus_remaining_equals_100", len(assigned) + len(remaining) == EXPECTED_PARTICIPANTS),
        ("assigned_and_remaining_disjoint", not (assigned_set & remaining_set)),
        ("assigned_remaining_union_is_expected_population",
         (assigned_set | remaining_set) == expected_ids),
        ("statement_assignment_counts_sum_to_79", sum(per_statement_assigned) == 79),
        ("participants_assigned_equals_79", len(assigned) == 79),
        ("participants_remaining_equals_21", len(remaining) == 21),
        ("selected_statements_equals_18", len(selected) == 18),
        ("budget_consumed_equals_359", selected_word_sum == 359),
        ("budget_remaining_equals_141", budget_remaining == 141),
        ("budget_consumed_plus_remaining_equals_500",
         selected_word_sum + budget_remaining == TOTAL_BUDGET),
        ("candidate_records_count_equals_candidate_number", len(records) == ckpt["candidate_number"]),
        ("all_scores_in_1_6", all(1 <= v <= 6 for v in all_scores)),
        ("score_cache_matches_records", cache_conflicts == 0),
        ("selected_statements_all_have_matching_generated_record",
         all(s["statement"] in stmt_to_records for s in selected)),
        ("selected_word_counts_match_count_words", budget["reconciliation"]["selected_word_counts_match_count_words"]),
    ]
    pd.DataFrame(
        [{"check": name, "passed": bool(ok)} for name, ok in checks]
    ).to_csv(OUT / "validation_checks.csv", index=False)
    all_pass = all(ok for _, ok in checks)

    # ------------------------------------------------------------------ #
    # Combined JSON + overview CSV                                        #
    # ------------------------------------------------------------------ #
    summary = {
        "evaluation_scope": "Partial (interrupted) 500-budget Algorithm 1 run - checkpoint snapshot only",
        "source_files": [str(CHECKPOINT.relative_to(ROOT)), str(MANIFEST.relative_to(ROOT))],
        "read_only_confirmation": "Original experiment files not modified",
        "run_metadata": {
            "model": manifest["model"],
            "endpoint": manifest["endpoint"],
            "participant_count": manifest["participant_count"],
            "seed": manifest["seed"],
            "status": manifest["budgets"][0]["status"],
            "termination_reason": ckpt["termination_reason"],
        },
        "extracted_counts": {
            "candidate_records": len(records),
            "successful_generations": len(successes),
            "generation_failures": len(failures),
            "selected_statements": len(selected),
            "assigned_participants": len(assigned),
            "remaining_participants": len(remaining),
            "score_cache_entries": len(score_cache),
        },
        "query_accuracy": query_acc,
        "representation": representation,
        "statement_diversity": diversity,
        "budget_utilisation": budget,
        "runtime": runtime,
        "validation": {"all_checks_passed": all_pass,
                       "checks": [{"check": n, "passed": bool(ok)} for n, ok in checks]},
        "limitations": [
            "Partial experiment: 18 of a potentially larger slate; 141 words unspent.",
            "No human ground truth: 'accuracy' is only model-score distribution/consistency.",
            "No statement embeddings and no Ollama calls: semantic similarity not computed.",
            "No per-event timestamps: separate generation/scoring times not available.",
            "Success/failure of generation reflects the 3-attempt retry cap and word-budget validation.",
        ],
    }
    (OUT / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    pd.DataFrame(
        [{"measure": "query_accuracy", "key": "scored_pairs_total", "value": len(all_scores)},
         {"measure": "query_accuracy", "key": "mean_score", "value": query_acc["score_summary"]["mean"]},
         {"measure": "representation", "key": "coverage_percent", "value": representation["coverage_percent"]},
         {"measure": "representation", "key": "participants_assigned", "value": len(assigned)},
         {"measure": "representation", "key": "statements_selected", "value": len(selected)},
         {"measure": "diversity", "key": "selected_unique_statements", "value": diversity["selected_unique_statements"]},
         {"measure": "diversity", "key": "mean_pairwise_jaccard", "value": diversity["lexical_jaccard_all_tokens"]["mean"]},
         {"measure": "diversity", "key": "semantic_similarity_computed", "value": False},
         {"measure": "budget", "key": "words_consumed", "value": selected_word_sum},
         {"measure": "budget", "key": "utilisation_percent", "value": budget["utilisation_percent"]},
         {"measure": "runtime", "key": "total_runtime_seconds", "value": total_runtime},
         {"measure": "runtime", "key": "per_stage_timings_available", "value": False}]
    ).to_csv(OUT / "metrics_overview.csv", index=False)

    # ------------------------------------------------------------------ #
    # Charts                                                              #
    # ------------------------------------------------------------------ #
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(OUT / name, dpi=150)
        plt.close(fig)

    # 1 Query accuracy
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(range(1, 7), [dist.get(k, 0) for k in range(1, 7)], color="#4472c4")
    ax[0].set(title="Model-based Disc approval score distribution", xlabel="Score (1-6)",
              ylabel="Count (scored pairs)")
    ax[1].hist(per_candidate_means, bins=20, color="#70ad47")
    ax[1].set(title="Per-candidate mean approval score", xlabel="Mean score", ylabel="Candidates")
    fig.suptitle("Query accuracy (LLM scores, NOT human ground truth)", fontsize=10)
    save(fig, "chart_1_query_accuracy.png")

    # 2 Representation
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(range(1, len(selected) + 1), per_statement_assigned, color="#ed7d31")
    ax[0].set(title="Participants assigned per selected statement", xlabel="Statement index",
              ylabel="Assigned participants")
    ax[1].pie([len(assigned), len(remaining)], labels=["Assigned", "Remaining"],
              autopct="%1.1f%%", colors=["#70ad47", "#d9d9d9"])
    ax[1].set(title="Participant coverage (79 / 100)")
    save(fig, "chart_2_representation.png")

    # 3 Diversity
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(jac, bins=20, color="#5b9bd5")
    ax[0].set(title="Pairwise token Jaccard (selected statements)", xlabel="Jaccard", ylabel="Pairs")
    ax[1].hist(lengths, bins=range(min(lengths), max(lengths) + 2), color="#a5a5a5")
    ax[1].set(title="Selected statement word counts", xlabel="Words", ylabel="Statements")
    fig.suptitle("Statement diversity (lexical only; semantic similarity unavailable)", fontsize=10)
    save(fig, "chart_3_diversity.png")

    # 4 Budget
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(["Consumed", "Remaining"], [selected_word_sum, budget_remaining],
              color=["#4472c4", "#d9d9d9"])
    ax[0].set(title=f"Budget utilisation = {budget['utilisation_percent']}%", ylabel="Words")
    cum = [s.get("budget_consumed") or 0 for s in selected]
    ax[1].plot(range(1, len(cum) + 1), cum, marker="o", color="#ed7d31")
    ax[1].set(title="Cumulative words consumed", xlabel="Selection index", ylabel="Words consumed")
    save(fig, "chart_4_budget.png")

    # 5 Runtime
    fig, ax = plt.subplots(figsize=(8, 3.5))
    derived = runtime["derived_aggregates_from_total_only"]
    numeric_keys = ["seconds_per_candidate_record", "seconds_per_successful_generation",
                    "seconds_per_selected_statement"]
    display_labels = ["sec / candidate record", "sec / successful generation",
                      "sec / selected statement"]
    vals = [derived[k] for k in numeric_keys]
    bars = ax.barh(display_labels, vals, color="#264478")
    ax.set_xlabel("Seconds (derived from total wall-clock only)")
    ax.set_title(f"Runtime: total {total_runtime:.0f}s until interrupt; no per-stage timings")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" {val:.1f}s", va="center")
    save(fig, "chart_5_runtime.png")

    print(f"Wrote evaluation artifacts to {OUT}")
    print(f"Validation all_pass={all_pass}; failed checks = "
          f"{[n for n, ok in checks if not ok]}")


if __name__ == "__main__":
    main()
