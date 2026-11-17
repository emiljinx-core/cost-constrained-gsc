import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from calibration.run_cost_constrained_experiment import (
    _all_generation_failures_are_connection_errors,
    _check_ollama_connectivity,
    run,
)
from PROSE.algorithm1_abortion import GenerationGroup


class _Response:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return self.payload


class _Agent:
    def get_id(self):
        return "generation1"

    def get_approval(self, statement):
        return 1, []


class CostConstrainedRunnerTests(unittest.TestCase):
    def test_connectivity_check_accepts_configured_model(self):
        with patch(
            "calibration.run_cost_constrained_experiment.urlopen",
            return_value=_Response({"models": [{"name": "qwen3:8b"}]}),
        ):
            _check_ollama_connectivity("http://ollama.test", "qwen3:8b", 1)

    def test_connectivity_check_rejects_missing_model(self):
        with patch(
            "calibration.run_cost_constrained_experiment.urlopen",
            return_value=_Response({"models": []}),
        ):
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                _check_ollama_connectivity("http://ollama.test", "qwen3:8b", 1)

    def test_only_connection_failures_mark_run_as_failed(self):
        connection_result = {
            "candidate_records": [
                {
                    "generation_success": False,
                    "generation_error": "OllamaProviderError: Could not reach Ollama",
                },
                {
                    "generation_success": False,
                    "generation_error": "OllamaHTTPError: server unavailable",
                },
            ]
        }
        format_result = {
            "candidate_records": [
                {
                    "generation_success": False,
                    "generation_error": "GenerationFormatError: missing opinion",
                }
            ]
        }
        self.assertTrue(_all_generation_failures_are_connection_errors(connection_result))
        self.assertFalse(_all_generation_failures_are_connection_errors(format_result))

    def test_keyboard_interrupt_marks_manifest_interrupted(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "calibration.run_cost_constrained_experiment._check_ollama_connectivity"
            ), patch(
                "calibration.run_cost_constrained_experiment._load_summaries",
                return_value={"generation1": "summary"},
            ), patch(
                "calibration.run_cost_constrained_experiment.load_nearest_generation_groups",
                return_value=[GenerationGroup("g", ("generation1",))],
            ), patch(
                "calibration.run_cost_constrained_experiment.load_abortion_agents",
                return_value=[_Agent()],
            ), patch(
                "calibration.run_cost_constrained_experiment.run_algorithm1",
                side_effect=KeyboardInterrupt,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    run(
                        budgets=[1],
                        output_dir=Path(directory),
                        endpoint="http://ollama.test",
                    )

            manifest = json.loads(
                (Path(directory) / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["budgets"][0]["status"], "interrupted")


if __name__ == "__main__":
    unittest.main()
