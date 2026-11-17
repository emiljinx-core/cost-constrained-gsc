"""Run the plain-generator abortion PROSE workflow with a local Ollama model."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from PROSE.datasets.abortion_adapter import load_abortion_agents
from PROSE.generate_scripts.generate_slate import generate_slate
from PROSE.queries.approval_scorers import OllamaApprovalScorer
from PROSE.queries.fast_queries import FastAgent, FastGenerator
from PROSE.utils.ollama_provider import OllamaProvider


DEFAULT_APPROVAL_LEVELS = [5.5, 5, 4.5, 4, 3.5, 3, 2, 1, 0]
DEFAULT_WORD_BUDGET_PER_AGENT = 2
DEFAULT_BUDGET_MULTIPLIERS = [
    40, 35, 30, 25, 20, 18, 16, 14, 12, 10, 8, 6, 5, 4, 3, 2, 1
]


def build_configuration(
    *,
    endpoint: str,
    model: str,
    word_budget: int | None = None,
    output_dir: Path | None = None,
    seed: int = 0,
    timeout: float = 120,
) -> dict[str, Any]:
    """Construct the local providers, agents, generator, and slate settings."""
    random.seed(seed)
    provider = OllamaProvider(
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

    agents = load_abortion_agents(
        model=model,
        specificity_coeff=0,
        agent_factory=agent_factory,
    )
    effective_word_budget = word_budget or (
        DEFAULT_WORD_BUDGET_PER_AGENT * len(agents)
    )
    words_per_agent = effective_word_budget // len(agents)
    if effective_word_budget <= 0 or effective_word_budget % len(agents) != 0:
        raise ValueError("word_budget must be a positive multiple of participant count")
    if words_per_agent < 1:
        raise ValueError("word_budget must allocate at least one word per participant")

    budget_schedule = [
        words_per_agent * multiplier
        for multiplier in DEFAULT_BUDGET_MULTIPLIERS
        if words_per_agent * multiplier <= effective_word_budget
    ]
    generator = FastGenerator(
        text_provider=provider,
        prompt_type="v3_very_high_detail",
        temperature=0,
    )
    return {
        "agents": agents,
        "precompute_generators": [],
        "generators": [generator],
        "word_budget": effective_word_budget,
        "budget_schedule": budget_schedule,
        "approval_levels": DEFAULT_APPROVAL_LEVELS.copy(),
        "full_approval_generation": False,
        "full_pool_selection": True,
        "num_threads": 1,
        "verbose": True,
        "no_budgets": False,
        "dataset": "abortion_ollama",
        "minimum_statement_length": 10,
        "output_dir": output_dir,
        "include_agent_data": False,
    }


def run(
    *,
    endpoint: str = "http://127.0.0.1:11434",
    model: str = "qwen3:8b",
    word_budget: int | None = None,
    output_dir: Path | None = None,
    seed: int = 0,
    timeout: float = 120,
):
    """Execute the local workflow through the unchanged slate-selection function."""
    configuration = build_configuration(
        endpoint=endpoint,
        model=model,
        word_budget=word_budget,
        output_dir=output_dir,
        seed=seed,
        timeout=timeout,
    )
    return generate_slate(**configuration)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--word-budget", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    result = run(
        endpoint=args.endpoint,
        model=args.model,
        word_budget=args.word_budget,
        output_dir=args.output_dir,
        seed=args.seed,
        timeout=args.timeout,
    )
    print(f"Done generating local abortion slate: {result[0]}")


if __name__ == "__main__":
    main()
