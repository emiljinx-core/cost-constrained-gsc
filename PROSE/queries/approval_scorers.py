import json
from typing import Any, List
from PROSE.utils.ollama_provider import ChatProvider


class OpenAIApprovalScorer:
    """Preserve the original probability-weighted OpenAI approval scoring."""

    supports_specificity = True

    def __init__(self, *, model: str, temperature: int):
        from PROSE.utils.llm_tools import GPT

        self.llm = GPT(
            temperature=temperature,
            model=model,
            logprobs=True,
            top_logprobs=5,
        )

    def score(
        self,
        *,
        data: str,
        statement: str,
        prompt_type: str,
        system_prompt: str,
        prompt: str,
    ) -> tuple[float, List[Any]]:
        from PROSE.utils.logprobs_tools import (
            filter_non_integer_idx,
            get_probabilities_from_completion,
        )

        response, completion, log = self.llm.call(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1,
        )

        try:
            logprobs = get_probabilities_from_completion(completion, 0)
            assert not logprobs.empty
            logprobs = filter_non_integer_idx(logprobs)
            assert not logprobs.empty
            score = float(
                (logprobs.astype(float) * logprobs.index.astype(float)).sum()
                / logprobs.astype(float).sum()
            )
        except AssertionError:
            print(
                f"Error: could not extract score from completion, response was {response}"
            )
            score = 0

        log["score"] = score
        return score, [log]


class OllamaApprovalScoringError(ValueError):
    """Raised when Ollama cannot return a valid approval score."""


class OllamaApprovalScorer:
    """Obtain a validated 1-6 agreement score from a chat provider."""

    supports_specificity = False

    def __init__(self, provider: ChatProvider, *, max_retries: int = 3):
        if max_retries < 1:
            raise ValueError("max_retries must be at least one")
        self.provider = provider
        self.max_retries = max_retries

    def score(
        self,
        *,
        data: str,
        statement: str,
        prompt_type: str,
        system_prompt: str,
        prompt: str,
    ) -> tuple[float, List[Any]]:
        scoring_system_prompt = (
            f"{system_prompt}\n\n"
            "Provider output contract: return exactly one JSON object with one "
            "field named score. The score must be an integer from 1 through 6. "
            "Do not include explanations, reasoning, markdown, or other fields."
        )
        scoring_prompt = (
            f"{prompt}\n\n"
            "Provider output contract: return only JSON in the form "
            '{"score": N}, where N is an integer from 1 through 6. '
            "Do not return any other text."
        )
        last_error = "no response was received"

        for _ in range(self.max_retries):
            response, _, log = self.provider.call(
                messages=[
                    {"role": "system", "content": scoring_system_prompt},
                    {"role": "user", "content": scoring_prompt},
                ],
                temperature=0,
                think=False,
                format="json",
                max_tokens=32,
            )
            try:
                score = self._parse_score(response)
            except OllamaApprovalScoringError as exc:
                last_error = str(exc)
                continue

            log["score"] = score
            return float(score), [log]

        raise OllamaApprovalScoringError(
            f"Ollama returned no valid approval score after {self.max_retries} attempts: "
            f"{last_error}"
        )

    @staticmethod
    def _parse_score(response: str) -> int:
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError as exc:
            raise OllamaApprovalScoringError("response is not valid JSON") from exc

        if not isinstance(parsed, dict) or set(parsed) != {"score"}:
            raise OllamaApprovalScoringError(
                "response must contain exactly the score field"
            )
        score = parsed["score"]
        if isinstance(score, bool) or not isinstance(score, int):
            raise OllamaApprovalScoringError("score must be an integer")
        if not 1 <= score <= 6:
            raise OllamaApprovalScoringError("score must be between 1 and 6")
        return score
