import json
from pathlib import Path
from tempfile import TemporaryDirectory
from io import StringIO
import unittest

from calibration.participant_summaries import (
    SUMMARY_SCHEMA_VERSION,
    SummaryValidationError,
    build_summary_messages,
    generate_summary,
    generate_participant_summaries,
    summary_to_embedding_text,
    validate_summary,
)


def valid_payload():
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "overall_summary": "The participant supports a conditional policy.",
        "policy_positions": [
            {
                "statement": "Supports access under stated conditions.",
                "evidence_response_indices": [1, 4],
            }
        ],
        "conditions_restrictions": [],
        "reasoning": [
            {
                "statement": "Explains the position using personal autonomy.",
                "evidence_response_indices": [2],
            }
        ],
        "contradictions_or_tensions": [],
        "missing_information": ["The responses do not state a preferred time limit."],
    }


class FakeProvider:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def call(self, messages, **params):
        self.calls.append((messages, params))
        return self.response, {}, {
            "provider": "ollama",
            "prompt": None,
            "response": None,
        }


class SequencedProvider(FakeProvider):
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def call(self, messages, **params):
        self.calls.append((messages, params))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response, {}, {"provider": "ollama", "prompt": None, "response": None}


class ParticipantSummaryTest(unittest.TestCase):
    responses = [f"Response text {index}" for index in range(1, 6)]

    def test_prompt_uses_all_responses_without_participant_id(self):
        messages = build_summary_messages(self.responses)
        prompt = messages[1]["content"]
        self.assertIn("Response 1:", prompt)
        self.assertIn("Response 5:", prompt)
        self.assertNotIn("generation1", prompt)

    def test_prompt_preserves_qualifiers_and_limits_missing_information(self):
        prompt = build_summary_messages(self.responses)[0]["content"]
        self.assertIn("qualified position", prompt)
        self.assertIn("policy-relevant information", prompt)
        self.assertIn("without restrictions", prompt)

    def test_missing_response_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "five response blocks"):
            build_summary_messages(self.responses[:4])

    def test_malformed_model_output_is_rejected(self):
        provider = FakeProvider("not-json")
        with self.assertRaises(SummaryValidationError):
            generate_summary(provider, self.responses)

    def test_contradictory_opinions_are_represented_with_evidence(self):
        payload = valid_payload()
        payload["contradictions_or_tensions"] = [
            {
                "statement": "Responses differ on whether exceptions are sufficient.",
                "evidence_response_indices": [1, 5],
            }
        ]
        summary, _ = generate_summary(
            FakeProvider(json.dumps(payload)),
            self.responses,
        )
        self.assertEqual(
            summary["contradictions_or_tensions"][0]["evidence_response_indices"],
            [1, 5],
        )

    def test_unsupported_claim_without_evidence_is_rejected(self):
        payload = valid_payload()
        payload["policy_positions"][0].pop("evidence_response_indices")
        with self.assertRaises(SummaryValidationError):
            validate_summary(payload)

    def test_invalid_evidence_index_is_rejected(self):
        payload = valid_payload()
        payload["reasoning"][0]["evidence_response_indices"] = [6]
        with self.assertRaises(SummaryValidationError):
            validate_summary(payload)

    def test_absolute_position_with_recorded_exception_is_qualified(self):
        payload = valid_payload()
        payload["policy_positions"][0]["statement"] = (
            "Abortion should be legal in all circumstances."
        )
        payload["contradictions_or_tensions"] = [
            {
                "statement": "The participant identifies a late-term exception.",
                "evidence_response_indices": [2, 4],
            }
        ]
        summary = validate_summary(payload)
        self.assertNotIn(
            "all circumstances",
            summary["policy_positions"][0]["statement"].lower(),
        )

    def test_irrelevant_missing_information_is_removed(self):
        payload = valid_payload()
        payload["missing_information"] = [
            "Specific demographic details about the participant"
        ]
        self.assertEqual(validate_summary(payload)["missing_information"], [])

    def test_embedding_text_is_stable_and_excludes_ids(self):
        text = summary_to_embedding_text(valid_payload())
        self.assertIn("conditional policy", text)
        self.assertIn("Missing information:", text)
        self.assertNotIn("generation1", text)

    def test_provider_call_disables_logging(self):
        provider = FakeProvider(json.dumps(valid_payload()))
        generate_summary(provider, self.responses)
        _, params = provider.calls[0]
        self.assertEqual(params["temperature"], 0)
        self.assertFalse(params["think"])
        self.assertEqual(params["format"], "json")

    def test_adapter_path_uses_data_only_factory(self):
        provider = FakeProvider(json.dumps(valid_payload()))
        with TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "summaries.json"
            records = generate_participant_summaries(
                provider=provider,
                participant_ids=["generation1"],
                output_path=output_path,
            )

        self.assertEqual([record["participant_id"] for record in records], ["generation1"])
        self.assertEqual(len(provider.calls), 1)

    def test_progress_and_checkpoint_after_success(self):
        provider = FakeProvider(json.dumps(valid_payload()))
        progress = StringIO()
        with TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "summaries.json"
            records = generate_participant_summaries(
                provider=provider,
                participant_ids=["generation1"],
                output_path=output_path,
                batch_number=2,
                progress_stream=progress,
            )
            checkpoint = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(len(records), 1)
        self.assertEqual(checkpoint["successful_count"], 1)
        self.assertEqual(checkpoint["failed_count"], 0)
        self.assertIn("MODEL=unknown TOTAL=1 batch=2", progress.getvalue())
        self.assertIn("generation1 PROCESSING", progress.getvalue())
        self.assertIn("generation1 SUCCESS", progress.getvalue())
        self.assertIn("PROCESSED=1 SUCCESSFUL=1 FAILED=0", progress.getvalue())

    def test_progress_continues_after_failure_and_checkpoints_failure(self):
        provider = SequencedProvider(
            [
                RuntimeError("simulated failure"),
                json.dumps(valid_payload()),
            ]
        )
        progress = StringIO()
        with TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "summaries.json"
            records = generate_participant_summaries(
                provider=provider,
                participant_ids=["generation1", "generation2"],
                output_path=output_path,
                progress_stream=progress,
            )
            checkpoint = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual([record["participant_id"] for record in records], ["generation2"])
        self.assertEqual(checkpoint["successful_count"], 1)
        self.assertEqual(checkpoint["failed_count"], 1)
        self.assertEqual(checkpoint["failures"][0]["participant_id"], "generation1")
        self.assertIn("generation1 FAILED", progress.getvalue())
        self.assertIn("generation2 SUCCESS", progress.getvalue())


if __name__ == "__main__":
    unittest.main()
