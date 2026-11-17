import json
import unittest
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from PROSE.generate_scripts import run_abortion_ollama
from PROSE.generate_scripts.generate_slate import generate_slate
from PROSE.queries.fast_queries import FastGenerator
from PROSE.utils.ollama_provider import OllamaProvider


class AbortionOllamaRunnerTest(unittest.TestCase):
    @staticmethod
    def _agents():
        return [object() for _ in range(100)]

    @patch("PROSE.generate_scripts.run_abortion_ollama.load_abortion_agents")
    def test_build_configuration_uses_injected_local_components(self, load_agents):
        load_agents.return_value = self._agents()

        configuration = run_abortion_ollama.build_configuration(
            endpoint="http://ollama.test:11434",
            model="qwen3:8b",
            word_budget=200,
            output_dir=Path("test-output"),
        )

        load_agents.assert_called_once()
        self.assertEqual(
            load_agents.call_args.kwargs["specificity_coeff"],
            0,
        )
        self.assertEqual(configuration["word_budget"], 200)
        self.assertEqual(configuration["output_dir"], Path("test-output"))
        self.assertFalse(configuration["include_agent_data"])
        self.assertEqual(configuration["approval_levels"], [5.5, 5, 4.5, 4, 3.5, 3, 2, 1, 0])
        self.assertEqual(len(configuration["generators"]), 1)
        self.assertIsInstance(configuration["generators"][0], FastGenerator)
        self.assertEqual(configuration["num_threads"], 1)

    @patch("PROSE.generate_scripts.run_abortion_ollama.load_abortion_agents")
    def test_configuration_shares_ollama_provider(self, load_agents):
        load_agents.return_value = self._agents()
        configuration = run_abortion_ollama.build_configuration(
            endpoint="http://ollama.test:11434",
            model="qwen3:8b",
            word_budget=200,
        )

        factory = load_agents.call_args.kwargs["agent_factory"]
        agent = factory(
            id="participant",
            data="private response",
            prompt_type="agreement_v1",
            specificity_prompt_type="specificity_v1",
            model="qwen3:8b",
            temperature=0,
            specificity_coeff=0,
            label="",
        )
        scorer_provider = agent.approval_scorer.provider
        self.assertIs(configuration["generators"][0].llm, scorer_provider)
        self.assertEqual(agent.specificity_coeff, 0)

    @patch("PROSE.generate_scripts.run_abortion_ollama.generate_slate")
    @patch("PROSE.generate_scripts.run_abortion_ollama.load_abortion_agents")
    def test_run_forwards_plain_generator_configuration(
        self, load_agents, generate_slate
    ):
        load_agents.return_value = self._agents()
        generate_slate.return_value = ([], [], [], [], Path("output"))

        result = run_abortion_ollama.run(
            endpoint="http://ollama.test:11434",
            model="qwen3:8b",
            word_budget=200,
            output_dir=Path("output"),
        )

        self.assertEqual(result[-1], Path("output"))
        forwarded = generate_slate.call_args.kwargs
        self.assertEqual(forwarded["dataset"], "abortion_ollama")
        self.assertEqual(len(forwarded["generators"]), 1)
        self.assertIsInstance(forwarded["generators"][0], FastGenerator)
        self.assertEqual(forwarded["output_dir"], Path("output"))
        self.assertFalse(forwarded["include_agent_data"])

    def test_runner_module_import_does_not_construct_openai(self):
        self.assertIsNotNone(run_abortion_ollama.build_configuration)

    def test_plain_selection_does_not_import_composites_or_log_data(self):
        class FakeAgent:
            def __init__(self, identifier):
                self.identifier = identifier

            def get_id(self):
                return self.identifier

            def get_approval(self, statement):
                return 1, [{"provider": "fake"}]

            def get_data(self):
                return "UNIQUE_PRIVATE_RESPONSE_MARKER"

            def get_label(self):
                return ""

        class FakeGenerator:
            uses_previous_statements = True

            def __init__(self):
                self.previous_statements = None

            def get_name(self):
                return "fake"

            def is_adaptable(self):
                return False

            def set_previous_statements(self, statements):
                self.previous_statements = statements

            def generate(self, agents, target_approval, target_budget):
                return ["shared policy"], []

        sys.modules.pop("PROSE.queries.gen_composite", None)
        generator = FakeGenerator()
        with tempfile.TemporaryDirectory() as temporary_directory:
            generate_slate(
                agents=[FakeAgent("1"), FakeAgent("2")],
                precompute_generators=[],
                generators=[generator],
                word_budget=2,
                budget_schedule=[2, 1],
                approval_levels=[5.5, 5, 4.5, 4, 3.5, 3, 2, 1, 0],
                num_threads=1,
                dataset="test",
                output_dir=Path(temporary_directory),
                include_agent_data=False,
            )
            self.assertNotIn("PROSE.queries.gen_composite", sys.modules)
            self.assertIsNotNone(generator.previous_statements)
            output = b"".join(
                path.read_bytes()
                for path in Path(temporary_directory).rglob("*")
                if path.is_file()
            )
            self.assertNotIn(b"UNIQUE_PRIVATE_RESPONSE_MARKER", output)

    @patch("PROSE.utils.ollama_provider.urlopen")
    def test_disabled_provider_logging_returns_no_prompt_or_response(self, urlopen):
        response = urlopen.return_value.__enter__.return_value
        response.read.return_value = json.dumps(
            {
                "model": "qwen3:8b",
                "message": {"role": "assistant", "content": "MODEL_RESPONSE_MARKER"},
            }
        ).encode("utf-8")
        provider = OllamaProvider(
            endpoint="http://ollama.test:11434",
            model="qwen3:8b",
            think=False,
            log_prompts=False,
            log_responses=False,
        )

        _, _, log = provider.call(
            [
                {
                    "role": "user",
                    "content": "PRIVATE_RESPONSE_MARKER",
                }
            ]
        )

        self.assertIsNone(log["prompt"])
        self.assertIsNone(log["system_prompt"])
        self.assertIsNone(log["messages"])
        self.assertIsNone(log["response"])
        self.assertNotIn("PRIVATE_RESPONSE_MARKER", repr(log))
        self.assertNotIn("MODEL_RESPONSE_MARKER", repr(log))


if __name__ == "__main__":
    unittest.main()
