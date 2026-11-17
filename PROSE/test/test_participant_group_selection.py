import unittest

import numpy as np

from calibration.participant_group_selection import (
    GroupSelectionError,
    balanced_kmeans,
    nearest_neighbor_groups,
    validate_embeddings,
)


class ParticipantGroupSelectionTests(unittest.TestCase):
    def setUp(self):
        self.ids = [f"generation{index}" for index in range(1, 11)]
        self.embeddings = np.eye(10)

    def test_balanced_kmeans_assigns_every_participant_once(self):
        result = balanced_kmeans(self.ids, self.embeddings, n_groups=5, random_seed=7)
        members = [participant_id for group in result["groups"] for participant_id in group["participant_ids"]]
        self.assertEqual(sorted(members), sorted(self.ids))
        self.assertEqual(len(members), len(set(members)))
        self.assertEqual([group["size"] for group in result["groups"]], [2] * 5)

    def test_uneven_balanced_capacities(self):
        result = balanced_kmeans(self.ids, self.embeddings, n_groups=3, random_seed=7)
        self.assertEqual([group["size"] for group in result["groups"]], [4, 3, 3])
        result = balanced_kmeans(self.ids, self.embeddings, group_size=4, random_seed=7)
        self.assertEqual([group["size"] for group in result["groups"]], [4, 4, 2])

    def test_nearest_neighbors_include_anchor_and_are_ordered(self):
        result = nearest_neighbor_groups(
            self.ids, self.embeddings, group_size=5, n_groups=2, random_seed=2
        )
        for group in result["groups"]:
            self.assertEqual(group["participant_ids"][0], group["anchor_id"])
            distances = [member["cosine_distance"] for member in group["members"]]
            self.assertEqual(distances, sorted(distances))

    def test_reproducibility_and_overlap(self):
        first = nearest_neighbor_groups(
            self.ids, self.embeddings, group_size=5, n_groups=2, random_seed=4
        )
        second = nearest_neighbor_groups(
            self.ids, self.embeddings, group_size=5, n_groups=2, random_seed=4
        )
        self.assertEqual(first, second)
        self.assertTrue(
            set(first["groups"][0]["participant_ids"])
            & set(first["groups"][1]["participant_ids"])
        )

    def test_different_experiments_use_independent_anchor_sequences(self):
        size_five = nearest_neighbor_groups(
            self.ids, self.embeddings, group_size=5, n_groups=2, random_seed=0
        )
        size_ten = nearest_neighbor_groups(
            self.ids, self.embeddings, group_size=10, n_groups=2, random_seed=1
        )
        self.assertNotEqual(
            [group["anchor_id"] for group in size_five["groups"]],
            [group["anchor_id"] for group in size_ten["groups"]],
        )

    def test_invalid_embeddings_and_sizes(self):
        with self.assertRaises(GroupSelectionError):
            validate_embeddings(["a", "a"], np.ones((2, 2)))
        with self.assertRaises(GroupSelectionError):
            validate_embeddings(["a", "b"], [[1.0, np.nan], [1.0, 0.0]])
        with self.assertRaises(GroupSelectionError):
            validate_embeddings(["a"], [[1.0, 0.0], [0.0, 1.0]])
        with self.assertRaises(GroupSelectionError):
            balanced_kmeans(self.ids, self.embeddings, n_groups=11)
        with self.assertRaises(GroupSelectionError):
            balanced_kmeans(self.ids, self.embeddings, n_groups=2, group_size=5)
        with self.assertRaises(GroupSelectionError):
            nearest_neighbor_groups(
                self.ids, self.embeddings, group_size=11, n_groups=1
            )

    def test_invalid_zero_vector_and_insufficient_participants(self):
        with self.assertRaises(GroupSelectionError):
            validate_embeddings(["a"], [[0.0, 0.0]])
        with self.assertRaises(GroupSelectionError):
            nearest_neighbor_groups(
                ["a", "b"], np.eye(2), group_size=3, n_groups=1
            )


if __name__ == "__main__":
    unittest.main()
