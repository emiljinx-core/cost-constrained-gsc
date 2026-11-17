"""Compare participant embedding similarity with calibration approval profiles."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.stats import rankdata, spearmanr


EXPECTED_IDS = tuple(f"generation{index}" for index in range(1, 101))
EMBEDDING_SCHEMA = "abortion-participant-embeddings-v1"
PROFILE_COLUMNS = ("human_rating", "qwen_score", "calibrated_score")


def participant_sort_key(participant_id: str) -> tuple[str, int]:
    prefix = participant_id.rstrip("0123456789")
    suffix = participant_id[len(prefix) :]
    return prefix, int(suffix or 0)


def load_embeddings(path: Path) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != EMBEDDING_SCHEMA:
        raise ValueError("Embedding file has an invalid schema")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != len(EXPECTED_IDS):
        raise ValueError("Embedding file must contain exactly 100 records")
    by_id: dict[str, list[float]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Embedding record must be an object")
        participant_id = record.get("participant_id")
        vector = record.get("embedding")
        if (
            not isinstance(participant_id, str)
            or participant_id in by_id
            or participant_id not in EXPECTED_IDS
            or not isinstance(vector, list)
            or len(vector) != int(payload.get("dimensions", 0))
        ):
            raise ValueError("Embedding records contain invalid IDs or vectors")
        numeric_vector = np.asarray(vector, dtype=float)
        if not np.isfinite(numeric_vector).all():
            raise ValueError(f"Embedding for {participant_id} is not finite")
        by_id[participant_id] = numeric_vector.tolist()
    if set(by_id) != set(EXPECTED_IDS):
        raise ValueError("Embedding IDs do not match generation1 through generation100")
    participant_ids = list(EXPECTED_IDS)
    matrix = np.asarray([by_id[participant_id] for participant_id in participant_ids])
    metadata = {
        key: payload.get(key)
        for key in (
            "schema_version",
            "model",
            "provider",
            "dimensions",
            "representation_version",
            "participant_count",
            "successful_count",
            "failed_count",
        )
    }
    return participant_ids, matrix, metadata


def load_calibration_profiles(
    path: Path, participant_ids: list[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    required = {"participant_id", "statement", *PROFILE_COLUMNS, "status"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Calibration CSV is missing columns: {sorted(required)}")
    participant_set = set(participant_ids)
    rows_by_participant: dict[str, list[dict[str, str]]] = {
        participant_id: [] for participant_id in participant_ids
    }
    for row in rows:
        participant_id = row["participant_id"]
        if participant_id in rows_by_participant:
            rows_by_participant[participant_id].append(row)
    if any(len(participant_rows) == 0 for participant_rows in rows_by_participant.values()):
        raise ValueError("At least one participant has no calibration rows")

    statement_order: list[str] = []
    for row in rows:
        statement = row["statement"]
        if statement not in statement_order:
            statement_order.append(statement)
    if len(statement_order) != 5:
        raise ValueError(f"Expected five distinct calibration statements, found {len(statement_order)}")

    profile_rows: list[dict[str, Any]] = []
    missing_counts = {column: 0 for column in PROFILE_COLUMNS}
    status_counts: dict[str, int] = {}
    for participant_id in participant_ids:
        participant_rows = rows_by_participant[participant_id]
        if len(participant_rows) != len(statement_order):
            raise ValueError(
                f"{participant_id} has {len(participant_rows)} rows; "
                f"expected {len(statement_order)}"
            )
        by_statement: dict[str, dict[str, str]] = {}
        for row in participant_rows:
            statement = row["statement"]
            if statement in by_statement:
                raise ValueError(f"Duplicate statement for {participant_id}")
            by_statement[statement] = row
            status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
        if set(by_statement) != set(statement_order):
            raise ValueError(f"Statement alignment mismatch for {participant_id}")
        profile: dict[str, Any] = {"participant_id": participant_id}
        for profile_column in PROFILE_COLUMNS:
            for index, statement in enumerate(statement_order, start=1):
                value = by_statement[statement].get(profile_column, "")
                key = f"{profile_column}__statement_{index}"
                if value == "":
                    profile[key] = np.nan
                    missing_counts[profile_column] += 1
                else:
                    try:
                        numeric_value = float(value)
                    except ValueError as exc:
                        raise ValueError(
                            f"Non-numeric {profile_column} for {participant_id}"
                        ) from exc
                    if not math.isfinite(numeric_value):
                        raise ValueError(f"Non-finite {profile_column} for {participant_id}")
                    profile[key] = numeric_value
        profile_rows.append(profile)
    metadata = {
        "row_count": len(rows),
        "participant_count": len(participant_ids),
        "statement_count": len(statement_order),
        "statements": statement_order,
        "missing_counts": missing_counts,
        "status_counts": status_counts,
        "participant_set_match": set(participant_ids) == participant_set,
    }
    return pd.DataFrame(profile_rows), metadata


def cosine_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0):
        raise ValueError("Cannot calculate cosine similarity for a zero vector")
    normalized = matrix / norms[:, None]
    return normalized @ normalized.T


def profile_similarity_matrix(values: np.ndarray) -> np.ndarray:
    similarity = np.full((len(values), len(values)), np.nan)
    for left in range(len(values)):
        for right in range(len(values)):
            mask = np.isfinite(values[left]) & np.isfinite(values[right])
            if not mask.any():
                continue
            left_vector = values[left, mask]
            right_vector = values[right, mask]
            left_norm = np.linalg.norm(left_vector)
            right_norm = np.linalg.norm(right_vector)
            if left_norm and right_norm:
                similarity[left, right] = float(
                    np.dot(left_vector, right_vector) / (left_norm * right_norm)
                )
    return similarity


def profile_mean_absolute_difference_matrix(values: np.ndarray) -> np.ndarray:
    differences = np.full((len(values), len(values)), np.nan)
    for left in range(len(values)):
        for right in range(len(values)):
            mask = np.isfinite(values[left]) & np.isfinite(values[right])
            if mask.any():
                differences[left, right] = float(
                    np.mean(np.abs(values[left, mask] - values[right, mask]))
                )
    return differences


def profile_pearson_matrix(values: np.ndarray) -> np.ndarray:
    correlations = np.full((len(values), len(values)), np.nan)
    for left in range(len(values)):
        for right in range(len(values)):
            mask = np.isfinite(values[left]) & np.isfinite(values[right])
            if mask.sum() < 2:
                continue
            left_vector = values[left, mask]
            right_vector = values[right, mask]
            value = pearson(left_vector, right_vector)
            if value is not None:
                correlations[left, right] = value
    return correlations


def pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def rank_values(values: np.ndarray) -> np.ndarray:
    return np.asarray(rankdata(values, method="average"), dtype=float)


def spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    result = spearmanr(left, right)
    value = float(result.statistic)
    return value if math.isfinite(value) else None


def pairwise_metrics(
    embedding_similarity: np.ndarray,
    profile_similarity: np.ndarray,
    profile_mean_absolute_difference: np.ndarray | None = None,
    profile_pearson: np.ndarray | None = None,
) -> dict[str, Any]:
    mask = np.triu(np.isfinite(profile_similarity), k=1)
    embedding_values = embedding_similarity[mask]
    profile_values = profile_similarity[mask]
    metrics = {
        "pair_count": int(mask.sum()),
        "pearson": pearson(embedding_values, profile_values),
        "spearman": spearman(embedding_values, profile_values),
        "embedding_similarity_mean": float(np.mean(embedding_values)),
        "profile_similarity_mean": float(np.mean(profile_values)),
    }
    if profile_mean_absolute_difference is not None:
        mad_mask = np.triu(np.isfinite(profile_mean_absolute_difference), k=1)
        mad_values = profile_mean_absolute_difference[mad_mask]
        negative_mad_values = -mad_values
        metrics.update(
            {
                "mean_absolute_difference_pair_count": int(mad_mask.sum()),
                "mean_absolute_difference_mean": float(np.mean(mad_values)),
                "pearson_vs_negative_mean_absolute_difference": pearson(
                    embedding_similarity[mad_mask], negative_mad_values
                ),
                "spearman_vs_negative_mean_absolute_difference": spearman(
                    embedding_similarity[mad_mask], negative_mad_values
                ),
            }
        )
    if profile_pearson is not None:
        correlation_mask = np.triu(np.isfinite(profile_pearson), k=1)
        correlation_values = profile_pearson[correlation_mask]
        metrics.update(
            {
                "profile_pearson_pair_count": int(correlation_mask.sum()),
                "profile_pearson_mean": float(np.mean(correlation_values))
                if len(correlation_values)
                else None,
                "pearson_vs_profile_pearson": pearson(
                    embedding_similarity[correlation_mask], correlation_values
                )
                if len(correlation_values)
                else None,
                "spearman_vs_profile_pearson": spearman(
                    embedding_similarity[correlation_mask], correlation_values
                )
                if len(correlation_values)
                else None,
            }
        )
    return metrics


def nearest_neighbors(
    participant_ids: list[str], similarity: np.ndarray, k: int = 5
) -> pd.DataFrame:
    rows = []
    for index, participant_id in enumerate(participant_ids):
        candidates = np.argsort(-similarity[index], kind="stable")
        neighbors = [candidate for candidate in candidates if candidate != index][:k]
        row: dict[str, Any] = {"participant_id": participant_id}
        for rank, neighbor in enumerate(neighbors, start=1):
            row[f"neighbor_{rank}"] = participant_ids[neighbor]
            row[f"similarity_{rank}"] = float(similarity[index, neighbor])
        rows.append(row)
    return pd.DataFrame(rows)


def save_heatmap(
    similarity: np.ndarray, participant_ids: list[str], path: Path
) -> None:
    figure, axis = plt.subplots(figsize=(13, 11))
    image = axis.imshow(similarity, cmap="viridis", vmin=-1, vmax=1, aspect="auto")
    axis.set_title("Participant embedding cosine similarity")
    axis.set_xlabel("Participant")
    axis.set_ylabel("Participant")
    ticks = np.arange(0, len(participant_ids), 10)
    axis.set_xticks(ticks, [participant_ids[index] for index in ticks], rotation=90)
    axis.set_yticks(ticks, [participant_ids[index] for index in ticks])
    figure.colorbar(image, ax=axis, label="Cosine similarity")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_pca_plot(matrix: np.ndarray, participant_ids: list[str], path: Path) -> None:
    coordinates = PCA(n_components=2, random_state=0).fit_transform(matrix)
    figure, axis = plt.subplots(figsize=(12, 9))
    axis.scatter(coordinates[:, 0], coordinates[:, 1], s=22)
    for index, participant_id in enumerate(participant_ids):
        axis.annotate(participant_id, (coordinates[index, 0], coordinates[index, 1]), fontsize=6)
    axis.set_title("PCA projection of participant embeddings")
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_comparison_plot(
    embedding_similarity: np.ndarray,
    profile_measures: dict[str, dict[str, np.ndarray]],
    path: Path,
) -> None:
    measure_labels = {
        "cosine": "Approval-profile cosine",
        "mean_absolute_difference": "Negative mean absolute difference",
        "profile_pearson": "Within-profile Pearson",
    }
    figure, axes = plt.subplots(
        len(profile_measures),
        len(measure_labels),
        figsize=(6 * len(measure_labels), 4.5 * len(profile_measures)),
        squeeze=False,
    )
    for row_index, (name, measures) in enumerate(profile_measures.items()):
        for column_index, (measure_name, measure_label) in enumerate(measure_labels.items()):
            axis = axes[row_index][column_index]
            profile_measure = measures[measure_name]
            profile_mask = np.triu(np.isfinite(profile_measure), k=1)
            y_values = profile_measure[profile_mask]
            if measure_name == "mean_absolute_difference":
                y_values = -y_values
            axis.scatter(
                embedding_similarity[profile_mask],
                y_values,
                s=5,
                alpha=0.35,
            )
            axis.set_title(f"{name}: {measure_label}")
            axis.set_xlabel("Embedding cosine similarity")
            axis.set_ylabel(measure_label)
            axis.set_xlim(-1, 1)
            if measure_name != "mean_absolute_difference":
                axis.set_ylim(-1, 1)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_analysis(
    *,
    embeddings_path: Path,
    calibration_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    participant_ids, embedding_matrix, embedding_metadata = load_embeddings(embeddings_path)
    profiles, calibration_metadata = load_calibration_profiles(
        calibration_path, participant_ids
    )
    embedding_similarity = cosine_similarity_matrix(embedding_matrix)
    profiles.to_csv(output_dir / "participant_approval_profiles.csv", index=False)
    pd.DataFrame(embedding_similarity, index=participant_ids, columns=participant_ids).to_csv(
        output_dir / "participant_embedding_cosine_similarity.csv"
    )
    neighbors = nearest_neighbors(participant_ids, embedding_similarity)
    neighbors.to_csv(output_dir / "participant_embedding_nearest_neighbors.csv", index=False)

    profile_measures: dict[str, dict[str, np.ndarray]] = {}
    for profile_column in PROFILE_COLUMNS:
        columns = [column for column in profiles if column.startswith(profile_column + "__")]
        values = profiles[columns].to_numpy(dtype=float)
        profile_measures[profile_column] = {
            "cosine": profile_similarity_matrix(values),
            "mean_absolute_difference": profile_mean_absolute_difference_matrix(values),
            "profile_pearson": profile_pearson_matrix(values),
        }
        pd.DataFrame(
            profile_measures[profile_column]["cosine"],
            index=participant_ids,
            columns=participant_ids,
        ).to_csv(output_dir / f"participant_{profile_column}_cosine_similarity.csv")
        pd.DataFrame(
            profile_measures[profile_column]["mean_absolute_difference"],
            index=participant_ids,
            columns=participant_ids,
        ).to_csv(
            output_dir
            / f"participant_{profile_column}_mean_absolute_difference.csv"
        )
        pd.DataFrame(
            profile_measures[profile_column]["profile_pearson"],
            index=participant_ids,
            columns=participant_ids,
        ).to_csv(
            output_dir / f"participant_{profile_column}_profile_pearson.csv"
        )

    metrics = {
        profile_column: pairwise_metrics(
            embedding_similarity,
            measures["cosine"],
            measures["mean_absolute_difference"],
            measures["profile_pearson"],
        )
        for profile_column, measures in profile_measures.items()
    }
    summary = {
        "embedding_file": str(embeddings_path),
        "calibration_file": str(calibration_path),
        "participant_ids": participant_ids,
        "embedding_metadata": embedding_metadata,
        "calibration_metadata": calibration_metadata,
        "profile_similarity_metrics": metrics,
        "outputs": {
            "embedding_similarity": "participant_embedding_cosine_similarity.csv",
            "nearest_neighbors": "participant_embedding_nearest_neighbors.csv",
            "approval_profiles": "participant_approval_profiles.csv",
            "human_similarity": "participant_human_rating_cosine_similarity.csv",
            "qwen_similarity": "participant_qwen_score_cosine_similarity.csv",
            "calibrated_similarity": "participant_calibrated_score_cosine_similarity.csv",
            "human_mean_absolute_difference": "participant_human_rating_mean_absolute_difference.csv",
            "qwen_mean_absolute_difference": "participant_qwen_score_mean_absolute_difference.csv",
            "calibrated_mean_absolute_difference": "participant_calibrated_score_mean_absolute_difference.csv",
            "human_profile_pearson": "participant_human_rating_profile_pearson.csv",
            "qwen_profile_pearson": "participant_qwen_score_profile_pearson.csv",
            "calibrated_profile_pearson": "participant_calibrated_score_profile_pearson.csv",
        },
    }
    (output_dir / "participant_embedding_similarity_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    save_heatmap(
        embedding_similarity,
        participant_ids,
        output_dir / "participant_embedding_cosine_heatmap.png",
    )
    save_pca_plot(
        embedding_matrix,
        participant_ids,
        output_dir / "participant_embedding_pca.png",
    )
    save_comparison_plot(
        embedding_similarity,
        profile_measures,
        output_dir / "embedding_vs_approval_similarity.png",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, default=Path("calibration/participant_embeddings_qwen3_0.6b.json"))
    parser.add_argument("--calibration", type=Path, default=Path("calibration/calibration_predictions.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("calibration/embedding_similarity_analysis"))
    args = parser.parse_args()
    summary = run_analysis(
        embeddings_path=args.embeddings,
        calibration_path=args.calibration,
        output_dir=args.output_dir,
    )
    print(
        f"Analysed {len(summary['participant_ids'])} participants; "
        f"results saved to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
