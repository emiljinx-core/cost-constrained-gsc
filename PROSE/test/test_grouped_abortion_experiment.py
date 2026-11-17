import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PROSE.generate_scripts.run_grouped_abortion_experiment import (
    ExperimentalRunnerError,
    _generate_candidate_with_retries,
    load_generation_groups,
    statement_similarity,
    summarize_scores,
)


class GroupedAbortionExperimentTests(unittest.TestCase):
    def test_summary_statistics_and_distribution(self):
        result = summarize_scores({"generation1": 1, "generation2": 3, "generation3": 3})
        self.assertEqual(result["mean"], 7 / 3)
        self.assertEqual(result["median"], 3)
        self.assertEqual(result["minimum"], 1)
        self.assertEqual(result["distribution"], {"1": 1, "3": 2})

    def test_statement_similarity_reports_duplicates(self):
        result = statement_similarity(["Legal access", "Legal access", "Different policy"])
        self.assertEqual(result["unique_statement_count"], 2)
        self.assertEqual(result["duplicate_count"], 1)

    def test_generation_retries_format_failures_but_never_accepts_malformed_output(self):
        class FakeGenerator:
            def __init__(self, responses):
                self.responses = iter(responses)

            def generate(self, agents, target_approval, target_budget):
                response = next(self.responses)
                if response == "malformed":
                    from PROSE.queries.fast_queries import GenerationFormatError

                    raise GenerationFormatError("malformed")
                return [response], []

        summaries = {"generation1": {"overall_summary": "supports access"}}
        statement, attempts = _generate_candidate_with_retries(
            FakeGenerator(["malformed", "<opinion>valid</opinion>"]),
            summaries,
            ["generation1"],
            target_budget=10,
            max_attempts=2,
        )
        self.assertEqual(statement, "<opinion>valid</opinion>")
        self.assertEqual(attempts, 2)

    def test_load_generation_groups_rejects_unknown_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            balanced = Path(directory) / "balanced.json"
            nearest = Path(directory) / "nearest.json"
            balanced.write_text(
                json.dumps({"groups": [{"group_index": 0, "participant_ids": ["unknown"]}]}),
                encoding="utf-8",
            )
            nearest.write_text(json.dumps({"experiments": []}), encoding="utf-8")
            with self.assertRaises(ExperimentalRunnerError):
                load_generation_groups(balanced, nearest, {"generation1"})

    @patch("PROSE.generate_scripts.run_grouped_abortion_experiment.FastGenerator")
    @patch("PROSE.generate_scripts.run_grouped_abortion_experiment.load_abortion_agents")
    @patch("PROSE.generate_scripts.run_grouped_abortion_experiment.load_generation_groups")
    @patch("PROSE.generate_scripts.run_grouped_abortion_experiment.load_validated_embeddings")
    @patch("PROSE.generate_scripts.run_grouped_abortion_experiment.load_summaries")
    def test_runner_uses_summary_for_generation_and_raw_agent_for_scoring(
        self, load_summaries, load_embeddings, load_groups, load_agents, generator_class
    ):
        class FakeAgent:
            def __init__(self, participant_id):
                self.participant_id = participant_id
                self.data = f"private response for {participant_id}"
                self.summary = None

            def get_id(self):
                return self.participant_id

            def get_approval(self, statement):
                self.scored_statement = statement
                return 4, []

        class FakeEmbeddings:
            participant_ids = ["generation1", "generation2"]

        class FakeGenerator:
            def __init__(self, **kwargs):
                self.prompt_agents = []

            def generate(self, agents, target_approval, target_budget):
                self.prompt_agents = agents
                return ["a generated policy statement"], []

        summaries = {
            "generation1": {"overall_summary": "supports access"},
            "generation2": {"overall_summary": "supports restrictions"},
        }
        agents = [FakeAgent("generation1"), FakeAgent("generation2")]
        load_summaries.return_value = summaries
        load_embeddings.return_value = FakeEmbeddings()
        load_groups.return_value = []
        load_agents.return_value = agents
        fake_generator = FakeGenerator()
        generator_class.return_value = fake_generator

        with tempfile.TemporaryDirectory() as directory:
            from PROSE.generate_scripts.run_grouped_abortion_experiment import run_experiment

            result = run_experiment(
                output_dir=Path(directory),
                provider_factory=lambda **kwargs: object(),
            )

        self.assertEqual(len(result["records"]), 1)
        self.assertTrue(result["records"][0]["generation"]["success"])
        self.assertEqual(
            result["records"][0]["evaluation"]["population_aggregate"]["mean"], 4
        )
        prompt_agents = fake_generator.prompt_agents
        self.assertNotIn("generation1", prompt_agents[0].get_id())
        self.assertIn("supports access", prompt_agents[0].get_description())
        self.assertEqual(
            [agent.scored_statement for agent in agents],
            ["a generated policy statement", "a generated policy statement"],
        )


if __name__ == "__main__":
    unittest.main()
