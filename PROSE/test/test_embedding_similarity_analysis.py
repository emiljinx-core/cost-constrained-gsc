import unittest

import numpy as np

from calibration.analyze_participant_embedding_similarity import (
    pairwise_metrics,
    profile_mean_absolute_difference_matrix,
    profile_pearson_matrix,
    profile_similarity_matrix,
    rank_values,
    spearman,
)


class EmbeddingSimilarityAnalysisTests(unittest.TestCase):
    def test_rank_values_uses_average_ranks_for_ties(self):
        np.testing.assert_array_equal(
            rank_values(np.array([1.0, 1.0, 2.0, 4.0])),
            np.array([1.5, 1.5, 3.0, 4.0]),
        )

    def test_spearman_handles_tied_values(self):
        self.assertAlmostEqual(
            spearman(
                np.array([0.1, 0.1, 0.4, 0.8]),
                np.array([0.2, 0.3, 0.3, 0.9]),
            ),
            0.833333333,
            places=7,
        )

    def test_constant_profiles_are_undefined_for_profile_pearson(self):
        values = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 2.0, 3.0]])
        cosine = profile_similarity_matrix(values)
        profile_pearson = profile_pearson_matrix(values)
        self.assertAlmostEqual(cosine[0, 1], 1.0)
        self.assertTrue(np.isnan(profile_pearson[0, 1]))
        self.assertTrue(np.isnan(profile_pearson[0, 2]))

    def test_missing_values_use_pairwise_overlap(self):
        values = np.array([[1.0, np.nan, 3.0], [1.0, 2.0, np.nan]])
        differences = profile_mean_absolute_difference_matrix(values)
        profile_pearson = profile_pearson_matrix(values)
        self.assertEqual(differences[0, 1], 0.0)
        self.assertTrue(np.isnan(profile_pearson[0, 1]))

    def test_pairwise_metrics_preserve_missing_pairs(self):
        embedding = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        profile = np.array([[1.0, np.nan, 3.0], [1.0, 2.0, np.nan], [1.0, 2.0, 3.0]])
        cosine = profile_similarity_matrix(profile)
        differences = profile_mean_absolute_difference_matrix(profile)
        within_profile = profile_pearson_matrix(profile)
        metrics = pairwise_metrics(embedding @ embedding.T, cosine, differences, within_profile)
        self.assertEqual(metrics["pair_count"], 3)
        self.assertEqual(metrics["mean_absolute_difference_pair_count"], 3)
        self.assertEqual(metrics["profile_pearson_pair_count"], 2)


if __name__ == "__main__":
    unittest.main()
