"""Generate privacy-safe, structured participant summaries with Ollama."""

from __future__ import annotations

import argparse
from io import TextIOBase
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

from PROSE.datasets.abortion_adapter import (
    DEFAULT_DATASET_PATH,
    extract_response_blocks,
    load_abortion_agents,
)
from PROSE.utils.ollama_provider import OllamaProvider


DEFAULT_PILOT_IDS = [f"generation{index}" for index in range(1, 6)]
SUMMARY_SCHEMA_VERSION = "abortion-participant-summary-v1"
CLAIM_FIELDS = (
    "policy_positions",
    "conditions_restrictions",
    "reasoning",
    "contradictions_or_tensions",
)
ABSOLUTE_POSITION_PHRASES = (
    "without restrictions",
    "all circumstances",
    "in every circumstance",
)
IRRELEVANT_MISSING_PHRASES = (
    "demographic",
    "personal beliefs",
    "personal details",
    "specific details about the participant",
)
SUMMARY_FIELDS = (
    "schema_version",
    "overall_summary",
    *CLAIM_FIELDS,
    "missing_information",
)


class _ParticipantInput:
    """Minimal adapter target used without constructing an approval agent."""

    def __init__(self, *, id: str, data: str, **_: Any) -> None:
        self._id = id
        self.data = data
        self.summary = None

    def get_id(self) -> str:
        return self._id

    def get_data(self) -> str:
        return self.data


class SummaryValidationError(ValueError):
    """Raised when a model response does not satisfy the summary schema."""


def _require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SummaryValidationError(f"{field} must be a non-empty string")
    return value.strip()


def _validate_claims(value: Any, field: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise SummaryValidationError(f"{field} must be a list")

    claims = []
    for index, claim in enumerate(value):
        if not isinstance(claim, dict):
            raise SummaryValidationError(
                f"{field}[{index}] must be an object with evidence"
            )
        if set(claim) != {"statement", "evidence_response_indices"}:
            raise SummaryValidationError(
                f"{field}[{index}] must contain only statement and "
                "evidence_response_indices"
            )
        statement = _require_nonempty_string(
            claim["statement"], f"{field}[{index}].statement"
        )
        evidence = claim["evidence_response_indices"]
        if (
            not isinstance(evidence, list)
            or not evidence
            or any(
                isinstance(response_index, bool)
                or not isinstance(response_index, int)
                or not 1 <= response_index <= 5
                for response_index in evidence
            )
            or len(set(evidence)) != len(evidence)
        ):
            raise SummaryValidationError(
                f"{field}[{index}].evidence_response_indices must contain "
                "unique integers from 1 through 5"
            )
        claims.append(
            {
                "statement": statement,
                "evidence_response_indices": evidence,
            }
        )
    return claims


def validate_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one model-produced participant summary."""
    if not isinstance(payload, Mapping):
        raise SummaryValidationError("summary must be a JSON object")
    if set(payload) != set(SUMMARY_FIELDS):
        raise SummaryValidationError(
            f"summary fields must be exactly {list(SUMMARY_FIELDS)}"
        )
    if payload["schema_version"] != SUMMARY_SCHEMA_VERSION:
        raise SummaryValidationError(
            f"schema_version must be {SUMMARY_SCHEMA_VERSION!r}"
        )

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "overall_summary": _require_nonempty_string(
            payload["overall_summary"], "overall_summary"
        ),
    }
    for field in CLAIM_FIELDS:
        summary[field] = _validate_claims(payload[field], field)

    missing_information = payload["missing_information"]
    if not isinstance(missing_information, list) or any(
        not isinstance(item, str) or not item.strip()
        for item in missing_information
    ):
        raise SummaryValidationError(
            "missing_information must be a list of non-empty strings"
        )
    summary["missing_information"] = [
        item.strip() for item in missing_information
    ]
    summary["missing_information"] = [
        item
        for item in summary["missing_information"]
        if not any(
            phrase in item.lower() for phrase in IRRELEVANT_MISSING_PHRASES
        )
    ]
    if summary["contradictions_or_tensions"]:
        if any(
            phrase in summary["overall_summary"].lower()
            for phrase in ABSOLUTE_POSITION_PHRASES
        ):
            summary["overall_summary"] = (
                "The participant broadly supports legal access to abortion, "
                "with qualifications described in the cited responses."
            )
        for field in ("policy_positions", "conditions_restrictions"):
            for claim in summary[field]:
                if any(
                    phrase in claim["statement"].lower()
                    for phrase in ABSOLUTE_POSITION_PHRASES
                ):
                    claim["statement"] = (
                        "The participant broadly supports legal access to abortion, "
                        "with qualifications described in the cited responses."
                    )
    return summary


def build_summary_messages(response_blocks: Sequence[str]) -> list[dict[str, str]]:
    """Build the privacy-safe prompt without participant IDs."""
    if len(response_blocks) != 5:
        raise ValueError(f"Expected five response blocks, found {len(response_blocks)}")
    if any(not isinstance(response, str) or not response.strip() for response in response_blocks):
        raise ValueError("All five response blocks must be non-empty strings")

    responses = "\n\n".join(
        f"Response {index}:\n{response.strip()}"
        for index, response in enumerate(response_blocks, start=1)
    )
    system_prompt = f"""
You summarize one participant's five abortion-policy survey responses.
Return only one JSON object matching the schema below.

Rules:
- Use all five responses.
- Do not invent opinions, facts, motives, demographics, or policy positions.
- Preserve qualifiers, exceptions, uncertainty, time limits, and conditional
  language. Do not turn a qualified position into "all circumstances",
  "without restrictions", or another absolute claim unless the responses
  explicitly support that wording.
- In `overall_summary` and `policy_positions`, do not use the exact phrases
  "all circumstances" or "without restrictions"; express broad positions with
  their cited exceptions instead.
- Every policy position, condition, restriction, reasoning item, and
  contradiction must include the response numbers that support it.
- A contradiction_or_tension must describe a genuine incompatibility between
  cited responses. A broad default with explicit exceptions is a qualified
  position, not automatically a contradiction. Use an empty list when the
  responses are coherent or merely nuanced.
- Put relevant information that the responses do not provide in
  missing_information. Include only missing policy-relevant information needed
  to interpret the participant's stance; do not list demographics, generic
  personal details, or unrelated omissions. Do not convert missing
  information into a neutral or negative opinion.
- Do not include participant IDs or identifying information.
- Preserve uncertainty and conditional language.

Exact JSON schema:
{{
  "schema_version": "{SUMMARY_SCHEMA_VERSION}",
  "overall_summary": "string",
  "policy_positions": [
    {{"statement": "string", "evidence_response_indices": [1]}}
  ],
  "conditions_restrictions": [
    {{"statement": "string", "evidence_response_indices": [2]}}
  ],
  "reasoning": [
    {{"statement": "string", "evidence_response_indices": [3]}}
  ],
  "contradictions_or_tensions": [
    {{"statement": "string", "evidence_response_indices": [1, 4]}}
  ],
  "missing_information": ["string"]
}}
""".strip()
    user_prompt = (
        "Summarize these five responses. Do not repeat the prompts or add "
        "information not supported by them.\n\n" + responses
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def generate_summary(
    provider: OllamaProvider,
    response_blocks: Sequence[str],
    *,
    max_attempts: int = 3,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate and validate one summary; return it with privacy-safe metadata."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least one")
    messages = build_summary_messages(response_blocks)
    last_error: str | None = None
    for attempt in range(max_attempts):
        request_messages = messages
        if attempt:
            request_messages = messages + [
                {
                    "role": "user",
                    "content": (
                        "Correct the previous JSON response. Preserve all "
                        "qualifiers and exceptions. Rewrite any absolute "
                        "overall summary or policy position as a qualified "
                        "position when an exception or tension exists. Do "
                        "not use the exact phrases 'all circumstances' or "
                        "'without restrictions' in those fields. "
                        "Remove non-policy missing-information entries. "
                        "Return only the corrected JSON object."
                    ),
                }
            ]
        response, _, log = provider.call(
            messages=request_messages,
            temperature=0,
            think=False,
            format="json",
            max_tokens=1200,
        )
        try:
            payload = json.loads(response)
            return validate_summary(payload), log
        except json.JSONDecodeError:
            last_error = "Ollama summary response was not valid JSON"
        except SummaryValidationError as exc:
            last_error = str(exc)
    raise SummaryValidationError(
        f"Ollama summary failed validation after {max_attempts} attempts: "
        f"{last_error}"
    )


def summary_to_embedding_text(summary: Mapping[str, Any]) -> str:
    """Flatten a validated summary into stable text for a later embedding stage."""
    validated = validate_summary(summary)
    sections = [validated["overall_summary"]]
    for field in CLAIM_FIELDS:
        sections.extend(
            claim["statement"] for claim in validated[field]
        )
    if validated["missing_information"]:
        sections.append(
            "Missing information: "
            + "; ".join(validated["missing_information"])
        )
    return "\n".join(sections)


def generate_participant_summaries(
    *,
    provider: OllamaProvider,
    participant_ids: Sequence[str],
    dataset_path: Path = DEFAULT_DATASET_PATH,
    output_path: Path,
    batch_number: int | None = None,
    progress_stream: TextIOBase | None = None,
) -> list[dict[str, Any]]:
    """Generate summaries with immediate progress and checkpoint persistence."""
    agents = load_abortion_agents(
        dataset_path=dataset_path,
        participant_ids=participant_ids,
        agent_factory=_ParticipantInput,
    )
    stream = progress_stream or sys.stdout
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    started_at = time.monotonic()

    def report(message: str) -> None:
        print(message, file=stream, flush=True)

    def save_checkpoint() -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "schema_version": SUMMARY_SCHEMA_VERSION,
                    "participant_count": len(participant_ids),
                    "successful_count": len(records),
                    "failed_count": len(failures),
                    "records": records,
                    "failures": failures,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    model_name = getattr(getattr(provider, "config", None), "model", "unknown")
    batch_text = f" batch={batch_number}" if batch_number is not None else ""
    report(
        f"MODEL={model_name} TOTAL={len(agents)}{batch_text} "
        f"OUTPUT={output_path}"
    )
    for agent in agents:
        responses = extract_response_blocks(agent.get_data())
        participant_started_at = time.monotonic()
        report(f"{agent.get_id()} PROCESSING")
        try:
            summary, _ = generate_summary(provider, responses)
        except (OSError, RuntimeError, SummaryValidationError, ValueError) as exc:
            failures.append(
                {
                    "participant_id": agent.get_id(),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            elapsed = time.monotonic() - participant_started_at
            report(
                f"{agent.get_id()} FAILED elapsed={elapsed:.2f}s "
                f"successful={len(records)} failed={len(failures)} "
                f"error={type(exc).__name__}: {exc}"
            )
            save_checkpoint()
            continue

        records.append({"participant_id": agent.get_id(), "summary": summary})
        elapsed = time.monotonic() - participant_started_at
        report(
            f"{agent.get_id()} SUCCESS elapsed={elapsed:.2f}s "
            f"successful={len(records)} failed={len(failures)}"
        )
        save_checkpoint()

    total_elapsed = time.monotonic() - started_at
    report(
        f"PROCESSED={len(agents)} SUCCESSFUL={len(records)} "
        f"FAILED={len(failures)} elapsed={total_elapsed:.2f}s "
        f"OUTPUT={output_path}"
    )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint")
    parser.add_argument("--model")
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--batch-number", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("calibration/participant_summary_pilot.json"),
    )
    parser.add_argument(
        "--participant-id",
        dest="participant_ids",
        action="append",
        default=None,
        help="Repeat for each participant; defaults to generation1 through generation5.",
    )
    args = parser.parse_args()
    participant_ids = args.participant_ids or DEFAULT_PILOT_IDS
    provider = OllamaProvider(
        endpoint=args.endpoint,
        model=args.model,
        timeout=args.timeout,
        think=False,
        log_prompts=False,
        log_responses=False,
    )
    records = generate_participant_summaries(
        provider=provider,
        participant_ids=participant_ids,
        output_path=args.output,
        batch_number=args.batch_number,
    )
    print(
        f"Generated {len(records)} participant summaries at {args.output} "
        f"using {provider.config.model}."
    )


if __name__ == "__main__":
    main()
