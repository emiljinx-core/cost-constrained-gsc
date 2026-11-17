"""Disc and Gen query contracts for the abortion-policy experiment."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Callable, Mapping, Sequence

from PROSE.queries.fast_queries import FastAgent, FastGenerator, GenerationFormatError
from PROSE.utils.helper_functions import count_words
from PROSE.queries.approval_scorers import OllamaApprovalScoringError
from PROSE.utils.ollama_provider import OllamaProvider, OllamaProviderError


class DiscQueryError(ValueError):
    """Raised when an abortion discriminative query cannot be completed."""


class GenQueryError(ValueError):
    """Raised when an abortion generative query cannot be completed."""


@dataclass(frozen=True)
class GeneratedStatement:
    """A validated Gen result and its measured word cost."""

    statement: str
    word_count: int
    target_approval: float
    word_budget: int
    participant_ids: tuple[str, ...]
    attempts: int


class AbortionDiscQuery:
    """Implement Disc(i, alpha) using the existing FastAgent scorer."""

    score_min = 1
    score_max = 6

    def __init__(self, agent: FastAgent):
        if not hasattr(agent, "get_approval"):
            raise DiscQueryError("agent must provide get_approval(statement)")
        self.agent = agent

    def score(self, statement: str) -> int:
        if not isinstance(statement, str) or not statement.strip():
            raise DiscQueryError("statement must be a non-empty string")
        try:
            score, _ = self.agent.get_approval(statement.strip())
        except (
            OllamaApprovalScoringError,
            OllamaProviderError,
            ValueError,
            RuntimeError,
        ) as exc:
            raise DiscQueryError(
                f"approval scoring failed: {type(exc).__name__}: {exc}"
            ) from exc
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise DiscQueryError("approval scorer returned a non-numeric score")
        if not math.isfinite(float(score)) or int(score) != score:
            raise DiscQueryError("approval scorer returned a non-integer score")
        integer_score = int(score)
        if not self.score_min <= integer_score <= self.score_max:
            raise DiscQueryError("approval score must be an integer from 1 through 6")
        return integer_score


class _SummaryPromptAgent:
    """Provide summary text to FastGenerator without exposing real IDs."""

    def __init__(self, label: str, description: str):
        self.label = label
        self.description = description

    def get_id(self) -> str:
        return self.label

    def get_description(self) -> str:
        return self.description


def _summary_text(summary: str | Mapping[str, Any]) -> str:
    if isinstance(summary, str):
        if not summary.strip():
            raise GenQueryError("participant summary must be non-empty")
        return summary.strip()
    if isinstance(summary, Mapping):
        return json.dumps(summary, sort_keys=True, ensure_ascii=True)
    raise GenQueryError("participant summary must be a string or mapping")


class AbortionGenQuery:
    """Implement Gen(S, l, x) using summary-only FastGenerator context."""

    def __init__(
        self,
        summaries: Mapping[str, str | Mapping[str, Any]],
        generator: FastGenerator,
        *,
        max_attempts: int = 3,
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    ):
        if max_attempts < 1:
            raise GenQueryError("max_attempts must be positive")
        if not summaries:
            raise GenQueryError("summaries must not be empty")
        self.summaries = {
            str(participant_id): _summary_text(summary)
            for participant_id, summary in summaries.items()
        }
        if any(not participant_id for participant_id in self.summaries):
            raise GenQueryError("participant IDs must be non-empty")
        self.generator = generator
        self.max_attempts = max_attempts
        self.progress_callback = progress_callback

    @classmethod
    def from_ollama(
        cls,
        summaries: Mapping[str, str | Mapping[str, Any]],
        *,
        endpoint: str | None = None,
        model: str = "qwen3:8b",
        timeout: float = 120,
        prompt_type: str = "v3_very_high_detail",
        temperature: float = 0,
        max_attempts: int = 3,
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> "AbortionGenQuery":
        provider = OllamaProvider(
            endpoint=endpoint,
            model=model,
            timeout=timeout,
            think=False,
            log_prompts=False,
            log_responses=False,
        )
        generator = FastGenerator(
            text_provider=provider,
            prompt_type=prompt_type,
            temperature=temperature,
        )
        return cls(
            summaries,
            generator,
            max_attempts=max_attempts,
            progress_callback=progress_callback,
        )

    def generate(
        self,
        participant_ids: Sequence[str],
        target_approval: float,
        word_budget: int,
    ) -> GeneratedStatement:
        ids = [str(participant_id) for participant_id in participant_ids]
        if not ids or len(ids) != len(set(ids)):
            raise GenQueryError("participant_ids must be a non-empty unique sequence")
        missing_ids = sorted(set(ids) - set(self.summaries))
        if missing_ids:
            raise GenQueryError(f"unknown participant IDs: {missing_ids}")
        if not isinstance(target_approval, (int, float)) or isinstance(
            target_approval, bool
        ):
            raise GenQueryError("target_approval must be numeric")
        if not math.isfinite(float(target_approval)):
            raise GenQueryError("target_approval must be finite")
        if not isinstance(word_budget, int) or isinstance(word_budget, bool) or word_budget <= 0:
            raise GenQueryError("word_budget must be a positive integer")

        prompt_agents = [
            _SummaryPromptAgent(
                f"participant_{index}",
                self.summaries[participant_id],
            )
            for index, participant_id in enumerate(ids, start=1)
        ]
        last_error: str | None = None
        for attempt in range(1, self.max_attempts + 1):
            if self.progress_callback is not None:
                self.progress_callback(
                    {
                        "event": "generation_attempt",
                        "attempt": attempt,
                        "target_approval": float(target_approval),
                        "word_budget": word_budget,
                        "participant_count": len(ids),
                    }
                )
            try:
                statements, _ = self.generator.generate(
                    prompt_agents,
                    target_approval=float(target_approval),
                    target_budget=word_budget,
                )
                if (
                    len(statements) != 1
                    or not isinstance(statements[0], str)
                    or not statements[0].strip()
                ):
                    raise GenQueryError(
                        "generator must return exactly one non-empty statement"
                    )
                statement = statements[0].strip()
                measured_words = count_words(statement)
                if measured_words > word_budget:
                    raise GenQueryError(
                        f"generated statement has {measured_words} words; "
                        f"word budget is {word_budget}"
                    )
                if self.progress_callback is not None:
                    self.progress_callback(
                        {
                            "event": "generation_result",
                            "attempt": attempt,
                            "success": True,
                            "word_count": measured_words,
                        }
                    )
                return GeneratedStatement(
                    statement=statement,
                    word_count=measured_words,
                    target_approval=float(target_approval),
                    word_budget=word_budget,
                    participant_ids=tuple(ids),
                    attempts=attempt,
                )
            except GenerationFormatError as exc:
                # Preserve the complete original reason (and its type) so the
                # progress output can distinguish XML/format, empty output and
                # other failures instead of reporting only "GenQueryError".
                last_error = f"{type(exc).__name__}: {exc}"
                if self.progress_callback is not None:
                    self.progress_callback(
                        {
                            "event": "generation_result",
                            "attempt": attempt,
                            "success": False,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        }
                    )
            except GenQueryError as exc:
                # Word-count validation and other GenQueryError reasons are
                # carried through verbatim, including the exception type.
                last_error = f"{type(exc).__name__}: {exc}"
                if self.progress_callback is not None:
                    self.progress_callback(
                        {
                            "event": "generation_result",
                            "attempt": attempt,
                            "success": False,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        }
                    )
        raise GenQueryError(
            f"generation failed after {self.max_attempts} attempts: {last_error}"
        )
