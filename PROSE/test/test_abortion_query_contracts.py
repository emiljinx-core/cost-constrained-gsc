import unittest
from unittest.mock import patch

from PROSE.queries.abortion_query_contracts import (
    AbortionDiscQuery,
    AbortionGenQuery,
    DiscQueryError,
    GenQueryError,
)
from PROSE.queries.fast_queries import GenerationFormatError
from PROSE.utils.ollama_provider import OllamaProviderError
from PROSE.utils.ollama_provider import OllamaProvider
from PROSE.queries.approval_scorers import OllamaApprovalScorer
from PROSE.queries.fast_queries import FastAgent, FastGenerator


class FakeAgent:
    def __init__(self, score=4, error=None):
        self.score_value = score
        self.error = error
        self.statements = []

    def get_approval(self, statement):
        self.statements.append(statement)
        if self.error:
            raise self.error
        return self.score_value, []


class FakeGenerator:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def generate(self, agents, target_approval, target_budget):
        self.calls.append(
            (
                [agent.get_id() for agent in agents],
                [agent.get_description() for agent in agents],
                target_approval,
                target_budget,
            )
        )
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        if response == "malformed":
            raise GenerationFormatError("malformed generation")
        return [response], []


class AbortionQueryContractTests(unittest.TestCase):
    def test_disc_scores_valid_statement_on_integer_ollama_scale(self):
        agent = FakeAgent(score=5.0)
        self.assertEqual(AbortionDiscQuery(agent).score("Legal access"), 5)
        self.assertEqual(agent.statements, ["Legal access"])

    def test_disc_rejects_empty_invalid_and_out_of_range_scores(self):
        query = AbortionDiscQuery(FakeAgent(score=4))
        for statement in ("", "   ", None):
            with self.subTest(statement=statement):
                with self.assertRaises(DiscQueryError):
                    query.score(statement)
        for score in (0, 7, 2.5):
            with self.subTest(score=score):
                with self.assertRaises(DiscQueryError):
                    AbortionDiscQuery(FakeAgent(score=score)).score("statement")

    def test_disc_surfaces_provider_failures(self):
        with self.assertRaisesRegex(DiscQueryError, "OllamaProviderError"):
            AbortionDiscQuery(
                FakeAgent(error=OllamaProviderError("connection failed"))
            ).score("statement")

    def test_gen_uses_subset_summaries_and_target(self):
        generator = FakeGenerator(["legal access"])
        query = AbortionGenQuery(
            {
                "generation1": {"overall_summary": "supports access"},
                "generation2": {"overall_summary": "supports limits"},
            },
            generator,
        )
        result = query.generate(["generation2"], target_approval=4.5, word_budget=5)
        self.assertEqual(result.statement, "legal access")
        self.assertEqual(result.word_count, 2)
        self.assertEqual(result.participant_ids, ("generation2",))
        self.assertEqual(generator.calls[0][0], ["participant_1"])
        self.assertIn("supports limits", generator.calls[0][1][0])
        self.assertEqual(generator.calls[0][2:], (4.5, 5))

    def test_gen_retries_over_budget_and_returns_measured_cost(self):
        generator = FakeGenerator(
            [
                "<opinion>one two three four</opinion>",
                "one two",
            ]
        )
        result = AbortionGenQuery(
            {"generation1": "supports access"}, generator, max_attempts=2
        ).generate(["generation1"], target_approval=4, word_budget=2)
        self.assertEqual(result.word_count, 2)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(len(generator.calls), 2)

    def test_gen_rejects_malformed_empty_and_exhausted_retries(self):
        for responses in (
            ["malformed", "malformed"],
            ["", ""],
        ):
            with self.subTest(responses=responses):
                with self.assertRaises(GenQueryError):
                    AbortionGenQuery(
                        {"generation1": "summary"},
                        FakeGenerator(responses),
                        max_attempts=2,
                    ).generate(["generation1"], target_approval=4, word_budget=5)

    def test_gen_rejects_invalid_subset_and_budget(self):
        query = AbortionGenQuery(
            {"generation1": "summary"},
            FakeGenerator(["<opinion>statement</opinion>"]),
        )
        with self.assertRaises(GenQueryError):
            query.generate(["missing"], target_approval=4, word_budget=5)
        with self.assertRaises(GenQueryError):
            query.generate(["generation1"], target_approval=4, word_budget=0)

    def test_disc_and_gen_use_the_same_environment_endpoint(self):
        endpoint = "http://172.21.112.1:11434"
        with patch.dict("os.environ", {"OLLAMA_BASE_URL": endpoint}):
            provider = OllamaProvider(model="qwen3:8b")
            scorer = OllamaApprovalScorer(provider)
            agent = FastAgent(
                id="generation1",
                data="response",
                prompt_type="agreement_v1",
                specificity_prompt_type="specificity_v1",
                approval_scorer=scorer,
                specificity_coeff=0,
            )
            disc = AbortionDiscQuery(agent)
            gen = AbortionGenQuery(
                {"generation1": "summary"},
                FastGenerator(text_provider=provider),
            )
            self.assertIs(disc.agent.approval_scorer.provider, gen.generator.llm)
            self.assertEqual(disc.agent.approval_scorer.provider.config.endpoint, endpoint)
            self.assertEqual(gen.generator.llm.config.endpoint, endpoint)

    def test_from_ollama_does_not_fall_back_to_loopback_endpoint(self):
        endpoint = "http://172.21.112.1:11434"
        with patch.dict("os.environ", {"OLLAMA_BASE_URL": endpoint}):
            query = AbortionGenQuery.from_ollama({"generation1": "summary"})
            self.assertEqual(query.generator.llm.config.endpoint, endpoint)

    def test_gen_reports_attempt_and_result_progress(self):
        events = []
        query = AbortionGenQuery(
            {"generation1": "summary"},
            FakeGenerator(["legal access"]),
            progress_callback=events.append,
        )
        query.generate(["generation1"], target_approval=4, word_budget=5)
        self.assertEqual(events[0]["event"], "generation_attempt")
        self.assertEqual(events[0]["attempt"], 1)
        self.assertEqual(events[1]["event"], "generation_result")
        self.assertTrue(events[1]["success"])


if __name__ == "__main__":
    unittest.main()
