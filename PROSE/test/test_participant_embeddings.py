import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from calibration.generate_participant_embeddings import (
    DEFAULT_DIMENSIONS,
    EMBEDDING_SCHEMA_VERSION,
    REPRESENTATION_VERSION,
    EmbeddingError,
    EmbeddingHTTPError,
    EmbeddingResponseError,
    OllamaEmbeddingClient,
    embedding_text,
    generate_embeddings,
    load_summary_dataset,
    validate_embedding,
    _write_output,
    EmbeddingConfig,
)
from calibration.participant_summaries import SUMMARY_SCHEMA_VERSION


def valid_summary():
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "overall_summary": "Supports a conditional policy.",
        "policy_positions": [
            {"statement": "Supports access.", "evidence_response_indices": [1]}
        ],
        "conditions_restrictions": [],
        "reasoning": [
            {"statement": "Values autonomy.", "evidence_response_indices": [2]}
        ],
        "contradictions_or_tensions": [],
        "missing_information": ["Unstated time limit"],
    }


def write_dataset(path: Path, participant_ids):
    path.write_text(
        json.dumps(
            {
                "schema_version": SUMMARY_SCHEMA_VERSION,
                "participant_count": len(participant_ids),
                "records": [
                    {"participant_id": participant_id, "summary": valid_summary()}
                    for participant_id in participant_ids
                ],
            }
        ),
        encoding="utf-8",
    )


class FakeHTTPResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return self.payload


class FakeClient:
    def __init__(self, responses):
        self.config = OllamaEmbeddingClient(
            endpoint="http://example.test",
            model="qwen3-embedding:0.6b",
            timeout=1,
        ).config
        self.responses = iter(responses)
        self.inputs = []

    def embed(self, text):
        self.inputs.append(text)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return validate_embedding(response)


class ParticipantEmbeddingTests(unittest.TestCase):
    vector = [0.1] * DEFAULT_DIMENSIONS

    def test_valid_embedding_response(self):
        client = OllamaEmbeddingClient(endpoint="http://example.test", timeout=1)
        with patch(
            "calibration.generate_participant_embeddings.urlopen",
            return_value=FakeHTTPResponse({"embeddings": [self.vector]}),
        ):
            self.assertEqual(client.embed("policy text"), self.vector)

    def test_vector_validation(self):
        self.assertEqual(len(validate_embedding(self.vector)), 1024)
        for bad in ([], [0.1], [float("nan")] * 1024, [float("inf")] * 1024):
            with self.assertRaises(EmbeddingResponseError):
                validate_embedding(bad)

    def test_invalid_transport_and_response_errors(self):
        client = OllamaEmbeddingClient(endpoint="http://example.test", timeout=1)
        with patch(
            "calibration.generate_participant_embeddings.urlopen",
            side_effect=URLError("offline"),
        ):
            with self.assertRaises(EmbeddingError):
                client.embed("policy text")
        with patch(
            "calibration.generate_participant_embeddings.urlopen",
            side_effect=HTTPError("url", 500, "error", {}, io.BytesIO(b"")),
        ):
            with self.assertRaises(EmbeddingHTTPError):
                client.embed("policy text")
        with patch(
            "calibration.generate_participant_embeddings.urlopen",
            return_value=FakeHTTPResponse({"not_embedding": []}),
        ):
            with self.assertRaises(EmbeddingResponseError):
                client.embed("policy text")
        with patch(
            "calibration.generate_participant_embeddings.urlopen",
            return_value=FakeHTTPResponse({"embedding": []}),
        ):
            with self.assertRaises(EmbeddingResponseError):
                client.embed("policy text")

    def test_embedding_text_excludes_missing_information(self):
        text = embedding_text(valid_summary())
        self.assertIn("Supports a conditional policy.", text)
        self.assertNotIn("Unstated time limit", text)

    def test_missing_and_duplicate_participant_ids(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "summaries.json"
            write_dataset(path, ["generation1"])
            with self.assertRaises(ValueError):
                load_summary_dataset(path, expected_participant_ids=["generation1", "generation2"])
            write_dataset(path, ["generation1", "generation1"])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_summary_dataset(path, expected_participant_ids=["generation1"])

    def test_atomic_checkpoint_replacement_and_cleanup_on_failure(self):
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "embeddings.json"
            config = EmbeddingConfig(
                endpoint="http://example.test",
                model="qwen3-embedding:0.6b",
                timeout=1,
            )
            _write_output(
                output_path,
                config=config,
                records=[],
                failures=[],
                participant_count=0,
            )
            self.assertTrue(output_path.exists())
            with patch.object(Path, "replace", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    _write_output(
                        output_path,
                        config=config,
                        records=[],
                        failures=[],
                        participant_count=1,
                    )
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])
            self.assertTrue(output_path.exists())

    def test_rejects_incompatible_representation_version(self):
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            input_path = directory / "summaries.json"
            output_path = directory / "embeddings.json"
            write_dataset(input_path, ["generation1"])
            output_path.write_text(
                json.dumps(
                    {
                        "schema_version": EMBEDDING_SCHEMA_VERSION,
                        "model": "qwen3-embedding:0.6b",
                        "provider": "ollama",
                        "dimensions": 1024,
                        "representation_version": "old-representation",
                        "records": [],
                        "failures": [],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "incompatible"):
                generate_embeddings(
                    client=FakeClient([]),
                    input_path=input_path,
                    output_path=output_path,
                    participant_ids=["generation1"],
                    expected_participant_ids=["generation1"],
                    progress_stream=io.StringIO(),
                )

    def test_resume_keeps_records_outside_requested_subset(self):
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            input_path = directory / "summaries.json"
            output_path = directory / "embeddings.json"
            write_dataset(input_path, ["generation1", "generation2"])
            first = FakeClient([self.vector])
            generate_embeddings(
                client=first,
                input_path=input_path,
                output_path=output_path,
                participant_ids=["generation2"],
                expected_participant_ids=["generation1", "generation2"],
                progress_stream=io.StringIO(),
            )
            second = FakeClient([self.vector])
            records, failures = generate_embeddings(
                client=second,
                input_path=input_path,
                output_path=output_path,
                participant_ids=["generation1"],
                expected_participant_ids=["generation1", "generation2"],
                progress_stream=io.StringIO(),
            )
            output = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual([record["participant_id"] for record in records], ["generation1", "generation2"])
        self.assertEqual([record["participant_id"] for record in output["records"]], ["generation1", "generation2"])
        self.assertEqual(failures, [])
        self.assertEqual(len(second.inputs), 1)

    def test_checkpoint_progress_failure_continuation_and_privacy(self):
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            input_path = directory / "summaries.json"
            output_path = directory / "embeddings.json"
            write_dataset(input_path, ["generation1", "generation2"])
            client = FakeClient([EmbeddingError("offline"), self.vector])
            progress = io.StringIO()
            records, failures = generate_embeddings(
                client=client,
                input_path=input_path,
                output_path=output_path,
                participant_ids=["generation1", "generation2"],
                expected_participant_ids=["generation1", "generation2"],
                progress_stream=progress,
            )
            output = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual([record["participant_id"] for record in records], ["generation2"])
        self.assertEqual(failures[0]["participant_id"], "generation1")
        self.assertIn("generation1 FAILED", progress.getvalue())
        self.assertIn("generation2 SUCCESS", progress.getvalue())
        self.assertNotIn("Supports a conditional policy.", json.dumps(output))
        self.assertEqual(output["schema_version"], EMBEDDING_SCHEMA_VERSION)

    def test_retry_failed_only(self):
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            input_path = directory / "summaries.json"
            output_path = directory / "embeddings.json"
            write_dataset(input_path, ["generation1", "generation2"])
            first = FakeClient([EmbeddingError("offline"), self.vector])
            generate_embeddings(
                client=first,
                input_path=input_path,
                output_path=output_path,
                participant_ids=["generation1", "generation2"],
                expected_participant_ids=["generation1", "generation2"],
                progress_stream=io.StringIO(),
            )
            second = FakeClient([self.vector])
            records, failures = generate_embeddings(
                client=second,
                input_path=input_path,
                output_path=output_path,
                participant_ids=["generation1", "generation2"],
                expected_participant_ids=["generation1", "generation2"],
                retry_failed=True,
                progress_stream=io.StringIO(),
            )
        self.assertEqual(len(records), 2)
        self.assertEqual(failures, [])
        self.assertEqual(len(second.inputs), 1)
        self.assertEqual(records[1]["embedding"], self.vector)


if __name__ == "__main__":
    unittest.main()
