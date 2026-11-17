"""Generate participant embeddings through Ollama's ``/api/embed`` endpoint."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence, TextIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from calibration.participant_summaries import (
    CLAIM_FIELDS,
    SUMMARY_SCHEMA_VERSION,
    summary_to_embedding_text,
    validate_summary,
)


DEFAULT_INPUT = Path("calibration/participant_summaries_all_100.json")
DEFAULT_OUTPUT = Path("calibration/participant_embeddings_qwen3_0.6b.json")
DEFAULT_MODEL = "qwen3-embedding:0.6b"
DEFAULT_DIMENSIONS = 1024
EMBEDDING_SCHEMA_VERSION = "abortion-participant-embeddings-v1"
REPRESENTATION_VERSION = "abortion-policy-summary-v1"
EXPECTED_PARTICIPANT_IDS = tuple(f"generation{index}" for index in range(1, 101))


class EmbeddingError(RuntimeError):
    """Base error for embedding transport or response failures."""


class EmbeddingHTTPError(EmbeddingError):
    """Raised when Ollama returns an HTTP error."""


class EmbeddingResponseError(EmbeddingError):
    """Raised when Ollama returns an unusable embedding response."""


def embedding_text(summary: Mapping[str, Any]) -> str:
    """Build deterministic primary representation text without missingness."""
    validated = validate_summary(summary)
    sections = [validated["overall_summary"]]
    for field in CLAIM_FIELDS:
        sections.extend(claim["statement"] for claim in validated[field])
    return "\n".join(section.strip() for section in sections if section.strip())


def validate_embedding(
    embedding: Any, *, dimensions: int = DEFAULT_DIMENSIONS
) -> list[float]:
    """Validate and normalize one embedding vector without logging its values."""
    if not isinstance(embedding, list) or not embedding:
        raise EmbeddingResponseError("Embedding must be a non-empty list")
    if len(embedding) != dimensions:
        raise EmbeddingResponseError(
            f"Embedding dimension must be {dimensions}, found {len(embedding)}"
        )
    validated: list[float] = []
    for index, value in enumerate(embedding):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EmbeddingResponseError(
                f"Embedding value at index {index} is not numeric"
            )
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise EmbeddingResponseError(
                f"Embedding value at index {index} is not finite"
            )
        validated.append(numeric_value)
    return validated


def load_summary_dataset(
    input_path: Path,
    *,
    expected_participant_ids: Sequence[str] = EXPECTED_PARTICIPANT_IDS,
) -> dict[str, dict[str, Any]]:
    """Load and validate the consolidated summary dataset."""
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load summary dataset: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != SUMMARY_SCHEMA_VERSION:
        raise ValueError("Summary dataset has an invalid schema_version")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("Summary dataset records must be a list")
    expected = list(expected_participant_ids)
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Summary dataset contains a non-object record")
        participant_id = record.get("participant_id")
        if not isinstance(participant_id, str) or not participant_id:
            raise ValueError("Summary dataset contains a missing participant ID")
        if participant_id in by_id:
            raise ValueError(f"Duplicate participant ID: {participant_id}")
        summary = record.get("summary")
        if not isinstance(summary, Mapping):
            raise ValueError(f"Missing summary for participant {participant_id}")
        validate_summary(summary)
        by_id[participant_id] = {"participant_id": participant_id, "summary": summary}
    if set(by_id) != set(expected) or len(by_id) != len(expected):
        missing = sorted(set(expected) - set(by_id))
        unexpected = sorted(set(by_id) - set(expected))
        raise ValueError(
            f"Participant IDs do not match expected dataset; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return by_id


@dataclass(frozen=True)
class EmbeddingConfig:
    endpoint: str
    model: str
    timeout: float
    dimensions: int = DEFAULT_DIMENSIONS

    @classmethod
    def from_environment(
        cls,
        *,
        endpoint: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        dimensions: int = DEFAULT_DIMENSIONS,
    ) -> "EmbeddingConfig":
        configured_timeout = timeout if timeout is not None else float(
            os.getenv("OLLAMA_TIMEOUT", "120")
        )
        if configured_timeout <= 0:
            raise ValueError("Ollama timeout must be greater than zero")
        configured_endpoint = endpoint or os.getenv(
            "OLLAMA_BASE_URL", "http://127.0.0.1:11434"
        )
        configured_model = model or DEFAULT_MODEL
        if not configured_endpoint:
            raise ValueError("Ollama endpoint must not be empty")
        if not configured_model:
            raise ValueError("Ollama model must not be empty")
        if dimensions <= 0:
            raise ValueError("Embedding dimensions must be positive")
        return cls(
            endpoint=configured_endpoint.rstrip("/"),
            model=configured_model,
            timeout=configured_timeout,
            dimensions=dimensions,
        )


class OllamaEmbeddingClient:
    """Minimal non-streaming client for Ollama's embedding endpoint."""

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        dimensions: int = DEFAULT_DIMENSIONS,
    ) -> None:
        self.config = EmbeddingConfig.from_environment(
            endpoint=endpoint,
            model=model,
            timeout=timeout,
            dimensions=dimensions,
        )

    def embed(self, text: str) -> list[float]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Embedding input must be a non-empty string")
        request = Request(
            f"{self.config.endpoint}/api/embed",
            data=json.dumps(
                {"model": self.config.model, "input": text}
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.config.timeout) as response:
                raw_body = response.read()
        except HTTPError as exc:
            raise EmbeddingHTTPError(
                f"Ollama returned HTTP {exc.code}"
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise EmbeddingError(
                f"Could not reach Ollama at {self.config.endpoint}: {type(exc).__name__}"
            ) from exc
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EmbeddingResponseError("Ollama returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise EmbeddingResponseError("Ollama embedding response must be an object")
        raw_embedding = payload.get("embedding")
        if raw_embedding is None:
            embeddings = payload.get("embeddings")
            if isinstance(embeddings, list) and len(embeddings) == 1:
                raw_embedding = embeddings[0]
        return validate_embedding(raw_embedding, dimensions=self.config.dimensions)


def _write_output(
    output_path: Path,
    *,
    config: EmbeddingConfig,
    records: Sequence[dict[str, Any]],
    failures: Sequence[dict[str, str]],
    participant_count: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": EMBEDDING_SCHEMA_VERSION,
        "model": config.model,
        "provider": "ollama",
        "dimensions": config.dimensions,
        "representation_version": REPRESENTATION_VERSION,
        "participant_count": participant_count,
        "successful_count": len(records),
        "failed_count": len(failures),
        "records": list(records),
        "failures": list(failures),
    }
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(payload, temporary_file, indent=2)
            temporary_file.write("\n")
            temporary_file.flush()
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _ordered_records(
    records: Mapping[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    order = {participant_id: index for index, participant_id in enumerate(
        EXPECTED_PARTICIPANT_IDS
    )}
    return [
        record
        for _, record in sorted(
            records.items(),
            key=lambda item: (order.get(item[0], len(order)), item[0]),
        )
    ]


def _load_existing(
    output_path: Path, *, config: EmbeddingConfig
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]]]:
    if not output_path.exists():
        return {}, {}
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load existing embedding output: {exc}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != EMBEDDING_SCHEMA_VERSION
        or payload.get("model") != config.model
        or payload.get("provider") != "ollama"
        or payload.get("dimensions") != config.dimensions
        or payload.get("representation_version") != REPRESENTATION_VERSION
    ):
        raise ValueError("Existing embedding output metadata is incompatible")
    raw_records = payload.get("records")
    raw_failures = payload.get("failures")
    if not isinstance(raw_records, list) or not isinstance(raw_failures, list):
        raise ValueError("Existing embedding output records and failures must be lists")
    records: dict[str, dict[str, Any]] = {}
    for record in raw_records:
        if not isinstance(record, dict):
            raise ValueError("Existing embedding record must be an object")
        participant_id = record.get("participant_id")
        if not isinstance(participant_id, str) or not participant_id:
            raise ValueError("Existing embedding record has an invalid participant ID")
        if participant_id in records:
            raise ValueError(f"Duplicate existing embedding ID: {participant_id}")
        vector = validate_embedding(record.get("embedding"), dimensions=config.dimensions)
        records[participant_id] = {"participant_id": participant_id, "embedding": vector}
    failures: dict[str, dict[str, str]] = {}
    for failure in raw_failures:
        if not isinstance(failure, dict):
            raise ValueError("Existing embedding failure must be an object")
        participant_id = failure.get("participant_id")
        if (
            not isinstance(participant_id, str)
            or not participant_id
            or not isinstance(failure.get("error_type"), str)
            or not isinstance(failure.get("error_message"), str)
        ):
            raise ValueError("Existing embedding failure has invalid fields")
        if participant_id in failures:
            raise ValueError(f"Duplicate existing embedding failure ID: {participant_id}")
        if participant_id in records:
            raise ValueError(f"Participant has both embedding and failure: {participant_id}")
        failures[participant_id] = {
            "participant_id": participant_id,
            "error_type": failure["error_type"],
            "error_message": failure["error_message"],
        }
    return records, failures


def generate_embeddings(
    *,
    client: OllamaEmbeddingClient,
    input_path: Path,
    output_path: Path,
    participant_ids: Sequence[str] = EXPECTED_PARTICIPANT_IDS,
    expected_participant_ids: Sequence[str] = EXPECTED_PARTICIPANT_IDS,
    retry_failed: bool = False,
    progress_stream: TextIO | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Generate requested embeddings, checkpointing after every participant."""
    summaries = load_summary_dataset(
        input_path, expected_participant_ids=expected_participant_ids
    )
    requested_ids = list(participant_ids)
    if len(requested_ids) != len(set(requested_ids)):
        raise ValueError("Duplicate participant IDs requested")
    unknown = sorted(set(requested_ids) - set(summaries))
    if unknown:
        raise ValueError(f"Unknown participant IDs requested: {unknown}")
    records, failures = _load_existing(output_path, config=client.config)
    stream = progress_stream or sys.stdout
    started_at = time.monotonic()

    def report(message: str) -> None:
        print(message, file=stream, flush=True)

    pending = [
        participant_id
        for participant_id in requested_ids
        if participant_id not in records
        and (retry_failed or participant_id not in failures)
    ]
    report(
        f"MODEL={client.config.model} TOTAL={len(requested_ids)} "
        f"OUTPUT={output_path}"
    )
    for participant_id in pending:
        participant_started_at = time.monotonic()
        report(f"{participant_id} PROCESSING")
        try:
            text = embedding_text(summaries[participant_id]["summary"])
            vector = client.embed(text)
            records[participant_id] = {
                "participant_id": participant_id,
                "embedding": vector,
            }
            failures.pop(participant_id, None)
            elapsed = time.monotonic() - participant_started_at
            report(
                f"{participant_id} SUCCESS elapsed={elapsed:.2f}s "
                f"successful={len(records)} failed={len(failures)}"
            )
        except (EmbeddingError, OSError, ValueError) as exc:
            failures[participant_id] = {
                "participant_id": participant_id,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            elapsed = time.monotonic() - participant_started_at
            report(
                f"{participant_id} FAILED elapsed={elapsed:.2f}s "
                f"successful={len(records)} failed={len(failures)} "
                f"error={type(exc).__name__}: {exc}"
            )
        _write_output(
            output_path,
            config=client.config,
            records=_ordered_records(records),
            failures=sorted(
                failures.values(), key=lambda item: item["participant_id"]
            ),
            participant_count=len(expected_participant_ids),
        )
    report(
        f"PROCESSED={len(pending)} SUCCESSFUL={len(records)} "
        f"FAILED={len(failures)} elapsed={time.monotonic() - started_at:.2f}s "
        f"OUTPUT={output_path}"
    )
    return _ordered_records(records), list(failures.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--endpoint")
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--participant-id", dest="participant_ids", action="append")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    participant_ids = args.participant_ids or EXPECTED_PARTICIPANT_IDS
    client = OllamaEmbeddingClient(
        endpoint=args.endpoint,
        model=args.model,
        timeout=args.timeout,
    )
    generate_embeddings(
        client=client,
        input_path=args.input,
        output_path=args.output,
        participant_ids=participant_ids,
        retry_failed=args.retry_failed,
    )


if __name__ == "__main__":
    main()
