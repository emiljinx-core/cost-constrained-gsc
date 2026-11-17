import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from calibration.run_calibration import (
    CalibrationMappingError,
    _apply_calibration,
    _apply_mapping,
    _isotonic_mapping,
    _write_outputs,
)


def _row(
    score: str, participant_id: str = "train", human_rating: str = "3"
) -> dict[str, str]:
    return {
        "participant_id": participant_id,
        "human_rating": human_rating,
        "qwen_score": score,
        "calibrated_score": "",
        "status": "success",
    }


class CalibrationMappingTest(unittest.TestCase):
    def test_observed_categories_are_mapped(self):
        rows = [
            _row("1"),
            _row("2"),
            _row("4"),
            _row("5"),
            _row("6"),
        ]

        _apply_calibration(rows, {"train"})

        self.assertTrue(all(row["calibrated_score"] != "" for row in rows))

    def test_unseen_category_fails_explicitly(self):
        rows = [
            _row("1", human_rating="0"),
            _row("2", human_rating="1"),
            _row("4", human_rating="2"),
            _row("5", human_rating="4"),
            _row("6", human_rating="5"),
            _row("3", participant_id="test"),
        ]

        with self.assertRaisesRegex(
            CalibrationMappingError,
            "No training calibration mapping exists",
        ):
            _apply_calibration(rows, {"train"})

        self.assertEqual(rows[-1]["calibrated_score"], "")

    def test_scores_outside_supported_range_fail(self):
        for score in ("0", "7"):
            with self.subTest(score=score):
                with self.assertRaisesRegex(
                    CalibrationMappingError,
                    "outside the supported range",
                ):
                    _apply_calibration([_row(score)], {"train"})

    def test_non_numeric_and_fractional_scores_fail(self):
        for score in ("not-a-number", "1.5"):
            with self.subTest(score=score):
                with self.assertRaisesRegex(
                    CalibrationMappingError,
                    "Invalid Qwen score",
                ):
                    _apply_calibration([_row(score)], {"train"})

    def test_empty_training_mapping_fails(self):
        with self.assertRaisesRegex(
            CalibrationMappingError,
            "No training calibration mapping exists",
        ):
            _apply_calibration([_row("1")], set())

    def test_failed_observations_are_left_uncalibrated(self):
        row = _row("3")
        row["status"] = "failed"
        row["error"] = "model failure"

        _apply_calibration([row], {"train"})

        self.assertEqual(row["calibrated_score"], "")

    def test_successful_empty_score_fails(self):
        with self.assertRaisesRegex(
            CalibrationMappingError,
            "missing a Qwen score",
        ):
            _apply_mapping([_row("")], {1: 0.46875})

    def test_failed_empty_score_remains_uncalibrated(self):
        row = _row("")
        row["status"] = "failed"

        _apply_mapping([row], {})

        self.assertEqual(row["calibrated_score"], "")

    def test_isotonic_mapping_known_training_outputs(self):
        rows = [
            _row("1", human_rating="0"),
            _row("2", human_rating="1"),
            _row("4", human_rating="2"),
            _row("5", human_rating="4"),
            _row("6", human_rating="5"),
        ]

        self.assertEqual(
            _isotonic_mapping(rows),
            {1: 0.0, 2: 1.0, 4: 2.0, 5: 4.0, 6: 5.0},
        )

    def test_output_path_enforces_mapping_validation(self):
        rows = [_row("3")]
        mapping = {1: 0.46875}

        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            with patch("calibration.run_calibration.PREDICTIONS_PATH", output_dir / "predictions.csv"), \
                patch("calibration.run_calibration.METRICS_PATH", output_dir / "metrics.json"), \
                patch("calibration.run_calibration.REPORT_PATH", output_dir / "report.md"):
                with self.assertRaisesRegex(
                    CalibrationMappingError,
                    "No training calibration mapping exists",
                ):
                    _write_outputs(
                        rows,
                        {"train"},
                        {"test"},
                        mapping=mapping,
                    )

    def test_observed_values_do_not_change_when_other_rows_are_present(self):
        rows = [
            _row("1", human_rating="0"),
            _row("2", human_rating="1"),
            _row("4", human_rating="2"),
            _row("5", human_rating="4"),
            _row("6", human_rating="5"),
        ]
        expected = {
            "1": 0,
            "2": 1,
            "4": 2,
            "5": 4,
            "6": 5,
        }

        _apply_calibration(rows, {"train"})

        for row in rows:
            self.assertAlmostEqual(
                float(row["calibrated_score"]),
                expected[row["qwen_score"]],
            )

    def test_existing_observed_mapping_values_are_preserved(self):
        rows = [_row(score) for score in ("1", "2", "4", "5", "6")]
        mapping = {
            1: 0.46875,
            2: 1.074468085106383,
            4: 2.6666666666666665,
            5: 3.8125,
            6: 3.9641025641025642,
        }

        _apply_mapping(rows, mapping)

        self.assertEqual(
            [float(row["calibrated_score"]) for row in rows],
            [mapping[int(row["qwen_score"])] for row in rows],
        )


if __name__ == "__main__":
    unittest.main()
