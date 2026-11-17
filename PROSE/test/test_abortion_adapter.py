import unittest
from pathlib import Path

import pandas as pd

from PROSE.datasets.abortion_adapter import (
    DEFAULT_DATASET_PATH,
    extract_response_blocks,
    load_abortion_agents,
)


class TestAgent:
    def __init__(self, *, id, data, **_):
        self.id = id
        self.data = data
        self.summary = None

    def get_id(self):
        return self.id

    def get_data(self):
        return self.data


class AbortionAdapterTest(unittest.TestCase):
    def test_loads_all_participants_without_llm_calls(self):
        agents = load_abortion_agents(agent_factory=TestAgent)

        dataset = pd.read_csv(
            Path(DEFAULT_DATASET_PATH),
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
        )
        ids = [agent.get_id() for agent in agents]
        response_counts = [len(extract_response_blocks(agent.get_data())) for agent in agents]

        self.assertEqual(len(agents), 100)
        self.assertEqual(ids[0], "generation1")
        self.assertEqual(ids[-1], "generation100")
        self.assertEqual(len(set(ids)), 100)
        self.assertEqual(response_counts, [5] * 100)
        self.assertTrue(all(agent.summary == agent.get_data() for agent in agents))
        self.assertEqual(
            {agent.get_id(): agent.get_data() for agent in agents},
            dict(zip(dataset["user_id"], dataset["data"])),
        )

        print(
            "Abortion adapter verification: "
            f"agents={len(agents)}, first_id={ids[0]}, last_id={ids[-1]}, "
            f"response_blocks={sorted(set(response_counts))}, "
            f"unique_ids={len(set(ids)) == len(ids)}, data_preserved=True"
        )

    def test_loads_requested_subset_in_requested_order(self):
        agents = load_abortion_agents(
            agent_factory=TestAgent,
            participant_ids=["generation2", "generation1"],
        )

        self.assertEqual(
            [agent.get_id() for agent in agents],
            ["generation2", "generation1"],
        )
        self.assertEqual(
            [len(extract_response_blocks(agent.get_data())) for agent in agents],
            [5, 5],
        )

    def test_rejects_duplicate_requested_ids(self):
        with self.assertRaisesRegex(ValueError, "Duplicate participant IDs"):
            load_abortion_agents(
                agent_factory=TestAgent,
                participant_ids=["generation1", "generation1"],
            )

    def test_rejects_unknown_requested_ids(self):
        with self.assertRaisesRegex(ValueError, "Unknown participant IDs"):
            load_abortion_agents(
                agent_factory=TestAgent,
                participant_ids=["generation1", "missing"],
            )


if __name__ == "__main__":
    unittest.main()
