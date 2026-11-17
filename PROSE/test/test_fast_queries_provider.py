import json
import importlib.util
import unittest

from PROSE.queries.approval_scorers import (
    OllamaApprovalScorer,
    OllamaApprovalScoringError,
)
from PROSE.queries.fast_queries import (
    FastAgent,
    FastGenerator,
    GenerationFormatError,
    StrictFastGenerator,
)


class FakeProvider:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def call(self, messages, **params):
        self.calls.append((messages, params))
        response = next(self.responses)
        return response, {}, {"provider": "fake"}


class FakeAgent:
    def __init__(self, identifier, description):
        self.identifier = identifier
        self.description = description

    def get_id(self):
        return self.identifier

    def get_description(self):
        return self.description


class ProviderIntegrationTest(unittest.TestCase):
    def test_ollama_scores_all_valid_integers(self):
        for expected in range(1, 7):
            scorer = OllamaApprovalScorer(
                FakeProvider([json.dumps({"score": expected})])
            )
            score, _ = scorer.score(
                data="participant",
                statement="statement",
                prompt_type="v1",
                system_prompt="unused",
                prompt="unused",
            )
            self.assertEqual(score, expected)

    def test_ollama_rejects_invalid_scores(self):
        invalid_responses = [
            "{}",
            '{"score": 1.5}',
            '{"score": 0}',
            '{"score": -1}',
            '{"score": 7}',
            '{"score": "3"}',
            "not json",
        ]
        for response in invalid_responses:
            with self.subTest(response=response):
                with self.assertRaises(OllamaApprovalScoringError):
                    OllamaApprovalScorer(
                        FakeProvider([response]), max_retries=1
                    ).score(
                        data="participant",
                        statement="statement",
                        prompt_type="v1",
                        system_prompt="unused",
                        prompt="unused",
                    )

    def test_ollama_retries_then_accepts_valid_score(self):
        provider = FakeProvider(["not json", '{"score": 4}'])
        scorer = OllamaApprovalScorer(provider, max_retries=2)
        score, _ = scorer.score(
            data="participant",
            statement="statement",
            prompt_type="v1",
            system_prompt="unused",
            prompt="unused",
        )
        self.assertEqual(score, 4)
        self.assertEqual(len(provider.calls), 2)

    def test_ollama_preserves_original_discriminative_prompt(self):
        provider = FakeProvider(['{"score": 3}'])
        scorer = OllamaApprovalScorer(provider, max_retries=1)
        scorer.score(
            data="participant",
            statement="statement",
            prompt_type="agreement_v1",
            system_prompt="ORIGINAL_SYSTEM_INSTRUCTION",
            prompt="ORIGINAL_USER_INSTRUCTION",
        )

        messages, params = provider.calls[0]
        self.assertIn("ORIGINAL_SYSTEM_INSTRUCTION", messages[0]["content"])
        self.assertIn("ORIGINAL_USER_INSTRUCTION", messages[1]["content"])
        self.assertIn("1 through 6", messages[0]["content"])
        self.assertIn("1 through 6", messages[1]["content"])
        self.assertEqual(params["temperature"], 0)
        self.assertFalse(params["think"])

    def test_injected_scorer_avoids_openai_and_disables_specificity(self):
        scorer = OllamaApprovalScorer(
            FakeProvider(['{"score": 5}'])
        )
        agent = FastAgent(
            id="user",
            data="participant",
            prompt_type="agreement_v1",
            specificity_prompt_type="specificity_v1",
            approval_scorer=scorer,
        )
        score, _ = agent.get_approval("statement")
        self.assertEqual(score, 5)
        self.assertFalse(agent.specificity_enabled)

    @unittest.skipUnless(
        all(
            importlib.util.find_spec(module) is not None
            for module in ("openai", "tenacity", "pydantic")
        ),
        "OpenAI dependencies are not installed",
    )
    def test_default_agent_constructs_openai_dependencies(self):
        agent = FastAgent(
            id="user",
            data="participant",
            prompt_type="agreement_v1",
            specificity_prompt_type="specificity_v1",
        )
        self.assertTrue(agent.specificity_enabled)

    def test_fast_generator_accepts_injected_provider(self):
        provider = FakeProvider(["<opinion>short statement</opinion>"])
        generator = FastGenerator(text_provider=provider)
        statements, _ = generator.generate(
            [FakeAgent("1", "description")], target_approval=4, target_budget=5
        )
        self.assertEqual(statements, ["short statement"])
        self.assertEqual(len(provider.calls), 1)

    def test_v3_very_high_detail_prompt_receives_target_approval(self):
        provider = FakeProvider(["<opinion>short statement</opinion>"])
        generator = FastGenerator(
            text_provider=provider, prompt_type="v3_very_high_detail"
        )
        statements, _ = generator.generate(
            [FakeAgent("1", "description")], target_approval=4.5, target_budget=7
        )
        system_prompt = provider.calls[0][0][0]["content"]
        self.assertEqual(statements, ["short statement"])
        self.assertIn("target approval level of 4.5", system_prompt)
        self.assertIn("at most 7 words", system_prompt)
        self.assertIn("VERY HIGH", system_prompt)
        self.assertIn("<opinion>...</opinion>", system_prompt)
        self.assertNotIn("{target_approval}", system_prompt)
        self.assertNotIn("{target_budget}", system_prompt)

    def test_fast_generator_rejects_missing_or_empty_opinion(self):
        for response, expected_message in [
            ("not tagged", "must contain"),
            ("<opinion>   </opinion>", "empty"),
            ("<opinion>one</opinion><opinion>two</opinion>", "exactly one"),
        ]:
            with self.subTest(response=response):
                generator = FastGenerator(
                    text_provider=FakeProvider([response])
                )
                with self.assertRaisesRegex(
                    GenerationFormatError, expected_message
                ):
                    generator.generate(
                        [FakeAgent("1", "description")],
                        target_approval=4,
                        target_budget=5,
                    )

    def test_fast_generator_rejects_malformed_opinion_block(self):
        generator = FastGenerator(
            text_provider=FakeProvider(
                ["prefix <opinion>statement</opinion> trailing"]
            )
        )
        with self.assertRaisesRegex(GenerationFormatError, "exactly one"):
            generator.generate(
                [FakeAgent("1", "description")],
                target_approval=4,
                target_budget=5,
            )

    def test_strict_generator_preserves_budget_retry(self):
        provider = FakeProvider(
            [
                "<opinion>one two three four</opinion>",
                "<opinion>one two</opinion>",
            ]
        )
        generator = StrictFastGenerator(
            text_provider=provider, num_retries=2
        )
        statements, _ = generator.generate(
            [FakeAgent("1", "description")], target_approval=4, target_budget=2
        )
        self.assertEqual(statements, ["one two"])
        self.assertEqual(len(provider.calls), 2)

    def test_injected_summary_provider_is_used(self):
        scorer = OllamaApprovalScorer(
            FakeProvider(['{"score": 5}'])
        )
        summary_provider = FakeProvider(['{"summary": "summary"}'])
        agent = FastAgent(
            id="user",
            data="participant",
            prompt_type="agreement_v1",
            specificity_prompt_type="specificity_v1",
            approval_scorer=scorer,
            summary_provider=summary_provider,
        )

        summary, _ = agent.generate_summary()

        self.assertEqual(summary, '{"summary": "summary"}')
        self.assertEqual(len(summary_provider.calls), 1)
        self.assertEqual(
            summary_provider.calls[0][1]["response_format"],
            {"type": "json_object"},
        )

if __name__ == "__main__":
    unittest.main()
