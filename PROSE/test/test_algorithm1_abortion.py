import unittest

from PROSE.algorithm1_abortion import (
    Algorithm1Error,
    GenerationGroup,
    required_support,
    run_algorithm1,
)
from PROSE.queries.abortion_query_contracts import GeneratedStatement, GenQueryError


class FakeDisc:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def score(self, statement):
        self.calls.append(statement)
        return self.scores.get(statement, 1)


class FakeGen:
    def __init__(self, statements):
        self.statements = statements
        self.calls = []

    def generate(self, participant_ids, target_approval, word_budget):
        self.calls.append((tuple(participant_ids), target_approval, word_budget))
        value = self.statements.get((target_approval, word_budget))
        if isinstance(value, BaseException):
            raise value
        statement = value or "x"
        return GeneratedStatement(
            statement=statement,
            word_count=len(statement.split()),
            target_approval=target_approval,
            word_budget=word_budget,
            participant_ids=tuple(participant_ids),
            attempts=1,
        )


class Algorithm1AbortionTests(unittest.TestCase):
    def test_required_support_uses_ceiling_cost_rule(self):
        self.assertEqual(required_support(1, 6, 10), 1)
        self.assertEqual(required_support(2, 6, 10), 2)
        self.assertEqual(required_support(10, 6, 10), 6)

    def test_algorithm_initialises_and_iterates_thresholds_and_costs(self):
        ids = ["p1", "p2", "p3"]
        gen = FakeGen({(3, 1): "x", (2, 1): "x", (1, 1): "x"})
        discs = {identifier: FakeDisc({"x": 1}) for identifier in ids}
        result = run_algorithm1(
            participant_ids=ids,
            disc_queries=discs,
            gen_query=gen,
            generation_groups=[GenerationGroup("g", tuple(ids))],
            total_budget=2,
            utility_levels=[3, 2, 1],
            cost_schedule=[1, 2],
        )
        self.assertEqual(result["initial_population_size"], 3)
        self.assertEqual(result["utility_levels"], [3, 2, 1])
        self.assertEqual(result["cost_schedule"], [1, 2])
        self.assertTrue(gen.calls)
        self.assertEqual(gen.calls[0][1:], (3, 1))

    def test_qualifies_assigns_highest_utility_and_removes_once(self):
        ids = ["p1", "p2", "p3", "p4"]
        gen = FakeGen({(2, 2): "one two"})
        discs = {
            "p1": FakeDisc({"one two": 6}),
            "p2": FakeDisc({"one two": 5}),
            "p3": FakeDisc({"one two": 4}),
            "p4": FakeDisc({"one two": 1}),
        }
        result = run_algorithm1(
            participant_ids=ids,
            disc_queries=discs,
            gen_query=gen,
            generation_groups=[GenerationGroup("g", tuple(ids))],
            total_budget=4,
            utility_levels=[2, 1],
            cost_schedule=[2],
        )
        self.assertEqual(
            result["selected_statements"][0]["assigned_participant_ids"],
            ["p1", "p2"],
        )
        self.assertEqual(
            result["assigned_participant_ids"],
            ["p1", "p2", "p3", "p4"],
        )
        self.assertEqual(result["budget_consumed"], 4)
        self.assertEqual(result["budget_remaining"], 0)

    def test_evaluates_against_remaining_population_and_allows_group_overlap(self):
        ids = ["p1", "p2", "p3"]
        gen = FakeGen({(2, 1): "x"})
        discs = {identifier: FakeDisc({"x": 6}) for identifier in ids}
        result = run_algorithm1(
            participant_ids=ids,
            disc_queries=discs,
            gen_query=gen,
            generation_groups=[
                GenerationGroup("g1", ("p1", "p2")),
                GenerationGroup("g2", ("p2", "p3")),
            ],
            total_budget=3,
            utility_levels=[2, 1],
            cost_schedule=[1],
        )
        self.assertEqual(result["selected_statements"][0]["assigned_participant_ids"], ["p1"])
        self.assertNotIn("p1", result["selected_statements"][0]["remaining_participant_ids"])
        self.assertEqual(len(gen.calls), 6)
        self.assertTrue(any("p2" in call[0] for call in gen.calls))

    def test_generation_failure_and_empty_candidates_do_not_break_run(self):
        ids = ["p1", "p2"]
        gen = FakeGen({(2, 1): GenQueryError("generation failed")})
        discs = {identifier: FakeDisc({}) for identifier in ids}
        result = run_algorithm1(
            participant_ids=ids,
            disc_queries=discs,
            gen_query=gen,
            generation_groups=[GenerationGroup("g", tuple(ids))],
            total_budget=1,
            utility_levels=[2],
            cost_schedule=[1],
        )
        self.assertEqual(result["selected_statements"], [])
        self.assertEqual(result["budget_consumed"], 0)
        self.assertTrue(
            any(record["generation_success"] is False for record in result["candidate_records"])
        )

    def test_budget_exhaustion_and_invalid_inputs(self):
        ids = ["p1", "p2"]
        gen = FakeGen({(2, 1): "x"})
        discs = {identifier: FakeDisc({"x": 6}) for identifier in ids}
        with self.assertRaises(Algorithm1Error):
            run_algorithm1(
                participant_ids=ids,
                disc_queries=discs,
                gen_query=gen,
                generation_groups=[GenerationGroup("g", tuple(ids))],
                total_budget=1,
                utility_levels=[2],
                cost_schedule=[2],
            )
        result = run_algorithm1(
            participant_ids=ids,
            disc_queries=discs,
            gen_query=gen,
            generation_groups=[GenerationGroup("g", tuple(ids))],
            total_budget=1,
            utility_levels=[2],
            cost_schedule=[1],
        )
        self.assertEqual(result["termination_reason"], "budget_exhausted")
        self.assertLessEqual(result["budget_consumed"], 1)

    def test_cost_schedule_exhaustion_is_distinct_from_budget_exhaustion(self):
        ids = ["p1", "p2"]
        gen = FakeGen({(2, 1): "x"})
        discs = {
            "p1": FakeDisc({"x": 2}),
            "p2": FakeDisc({"x": 1}),
        }
        result = run_algorithm1(
            participant_ids=ids,
            disc_queries=discs,
            gen_query=gen,
            generation_groups=[GenerationGroup("g", tuple(ids))],
            total_budget=3,
            utility_levels=[2],
            cost_schedule=[1, 2, 3],
        )
        self.assertEqual(result["budget_remaining"], 2)
        self.assertEqual(result["termination_reason"], "cost_schedule_exhausted")

    def test_progress_callback_reports_generation_scoring_and_selection(self):
        events = []
        ids = ["p1", "p2"]
        gen = FakeGen({(2, 1): "x"})
        discs = {identifier: FakeDisc({"x": 2}) for identifier in ids}
        run_algorithm1(
            participant_ids=ids,
            disc_queries=discs,
            gen_query=gen,
            generation_groups=[GenerationGroup("g", tuple(ids))],
            total_budget=2,
            utility_levels=[2],
            cost_schedule=[1],
            progress_callback=events.append,
        )
        event_names = [event["event"] for event in events]
        self.assertIn("generation_request", event_names)
        self.assertIn("disc_scoring_completed", event_names)
        self.assertIn("statement_selected", event_names)
        self.assertEqual(event_names[-1], "algorithm_completed")


if __name__ == "__main__":
    unittest.main()
