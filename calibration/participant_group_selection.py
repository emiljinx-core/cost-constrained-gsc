"""Independent participant grouping strategies for embedding diagnostics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from calibration.analyze_participant_embedding_similarity import load_embeddings


DEFAULT_EMBEDDINGS = Path("calibration/participant_embeddings_qwen3_0.6b.json")
DEFAULT_BALANCED_OUTPUT = Path("calibration/balanced_kmeans_groups.json")
DEFAULT_NEAREST_OUTPUT = Path("calibration/nearest_neighbor_groups.json")


class GroupSelectionError(ValueError):
    """Raised when participant group selection inputs are invalid."""


@dataclass(frozen=True)
class ValidatedEmbeddings:
    participant_ids: list[str]
    vectors: np.ndarray
    metadata: dict[str, Any]


def validate_embeddings(
    participant_ids: Sequence[str], embeddings: Any
) -> ValidatedEmbeddings:
    ids = list(participant_ids)
    if not ids or len(ids) != len(set(ids)) or any(
        not isinstance(participant_id, str) or not participant_id for participant_id in ids
    ):
        raise GroupSelectionError("Participant IDs must be unique non-empty strings")
    vectors = np.asarray(embeddings, dtype=float)
    if vectors.ndim != 2 or vectors.shape[0] != len(ids) or vectors.shape[1] == 0:
        raise GroupSelectionError("Embeddings must be a non-empty 2D matrix aligned to IDs")
    if not np.isfinite(vectors).all():
        raise GroupSelectionError("Embeddings must contain only finite values")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms == 0):
        raise GroupSelectionError("Embeddings must not contain zero vectors")
    return ValidatedEmbeddings(ids, vectors / norms[:, None], {})


def load_validated_embeddings(path: Path = DEFAULT_EMBEDDINGS) -> ValidatedEmbeddings:
    participant_ids, matrix, metadata = load_embeddings(path)
    validated = validate_embeddings(participant_ids, matrix)
    return ValidatedEmbeddings(
        validated.participant_ids, validated.vectors, metadata
    )


def _resolve_capacities(
    participant_count: int,
    *,
    n_groups: int | None,
    group_size: int | None,
) -> list[int]:
    if (n_groups is None) == (group_size is None):
        raise GroupSelectionError("Specify exactly one of n_groups or group_size")
    if n_groups is not None:
        if n_groups <= 0 or n_groups > participant_count:
            raise GroupSelectionError("n_groups must be between 1 and participant count")
        base, remainder = divmod(participant_count, n_groups)
        return [base + (index < remainder) for index in range(n_groups)]
    assert group_size is not None
    if group_size <= 0 or group_size > participant_count:
        raise GroupSelectionError("group_size must be between 1 and participant count")
    n_groups, remainder = divmod(participant_count, group_size)
    capacities = [group_size] * n_groups
    if remainder:
        capacities.append(remainder)
    return capacities


def balanced_kmeans(
    participant_ids: Sequence[str],
    embeddings: Any,
    *,
    n_groups: int | None = None,
    group_size: int | None = None,
    random_seed: int = 0,
    max_iter: int = 100,
) -> dict[str, Any]:
    """Assign every participant to fixed-capacity clusters.

    Centroids are initialized reproducibly. Each iteration solves a linear
    assignment problem over repeated centroid slots, which enforces the
    requested capacities exactly.
    """
    validated = validate_embeddings(participant_ids, embeddings)
    if max_iter <= 0:
        raise GroupSelectionError("max_iter must be positive")
    capacities = _resolve_capacities(
        len(validated.participant_ids), n_groups=n_groups, group_size=group_size
    )
    rng = np.random.default_rng(random_seed)
    centroid_indices = rng.choice(
        len(validated.participant_ids), size=len(capacities), replace=False
    )
    centroids = validated.vectors[centroid_indices].copy()
    assignments = np.full(len(validated.participant_ids), -1, dtype=int)
    iterations = 0

    for iterations in range(1, max_iter + 1):
        distances = np.sum(
            (validated.vectors[:, None, :] - centroids[None, :, :]) ** 2,
            axis=2,
        )
        slot_centroids = np.repeat(np.arange(len(capacities)), capacities)
        slot_costs = distances[:, slot_centroids]
        rows, slots = linear_sum_assignment(slot_costs)
        new_assignments = np.full(len(validated.participant_ids), -1, dtype=int)
        new_assignments[rows] = slot_centroids[slots]
        new_centroids = centroids.copy()
        for cluster_index in range(len(capacities)):
            members = validated.vectors[new_assignments == cluster_index]
            new_centroids[cluster_index] = members.mean(axis=0)
            norm = np.linalg.norm(new_centroids[cluster_index])
            if norm:
                new_centroids[cluster_index] /= norm
        if np.array_equal(assignments, new_assignments):
            assignments = new_assignments
            centroids = new_centroids
            break
        assignments = new_assignments
        centroids = new_centroids

    groups = []
    for cluster_index, capacity in enumerate(capacities):
        member_indices = np.flatnonzero(assignments == cluster_index)
        groups.append(
            {
                "group_index": cluster_index,
                "participant_ids": [
                    validated.participant_ids[index] for index in member_indices
                ],
                "size": len(member_indices),
                "capacity": capacity,
            }
        )
    return {
        "strategy": "balanced_kmeans",
        "configuration": {
            "n_groups": len(capacities),
            "requested_group_size": group_size,
            "capacities": capacities,
            "random_seed": random_seed,
            "max_iter": max_iter,
            "iterations": iterations,
            "distance": "squared_euclidean_on_unit_normalized_embeddings",
        },
        "embedding_metadata": validated.metadata,
        "groups": groups,
    }


def nearest_neighbor_groups(
    participant_ids: Sequence[str],
    embeddings: Any,
    *,
    group_size: int,
    n_groups: int,
    random_seed: int = 0,
) -> dict[str, Any]:
    """Select independently sampled random-anchor nearest-neighbour groups."""
    validated = validate_embeddings(participant_ids, embeddings)
    if group_size <= 0 or group_size > len(validated.participant_ids):
        raise GroupSelectionError("group_size must be between 1 and participant count")
    if n_groups <= 0:
        raise GroupSelectionError("n_groups must be positive")
    similarity = validated.vectors @ validated.vectors.T
    distances = 1.0 - similarity
    rng = random.Random(random_seed)
    groups = []
    for group_index in range(n_groups):
        anchor_index = rng.randrange(len(validated.participant_ids))
        ordered_indices = sorted(
            (index for index in range(len(validated.participant_ids)) if index != anchor_index),
            key=lambda index: (distances[anchor_index, index], validated.participant_ids[index]),
        )
        selected_indices = [anchor_index] + ordered_indices[: group_size - 1]
        selected_indices = [anchor_index] + sorted(
            selected_indices[1:],
            key=lambda index: (distances[anchor_index, index], validated.participant_ids[index]),
        )
        groups.append(
            {
                "group_index": group_index,
                "anchor_id": validated.participant_ids[anchor_index],
                "participant_ids": [
                    validated.participant_ids[index] for index in selected_indices
                ],
                "size": len(selected_indices),
                "members": [
                    {
                        "participant_id": validated.participant_ids[index],
                        "cosine_distance": float(distances[anchor_index, index]),
                    }
                    for index in selected_indices
                ],
            }
        )
    return {
        "strategy": "nearest_neighbor",
        "configuration": {
            "group_size": group_size,
            "n_groups": n_groups,
            "random_seed": random_seed,
            "distance": "cosine_distance",
            "overlap_allowed": True,
        },
        "embedding_metadata": validated.metadata,
        "groups": groups,
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--balanced-output", type=Path, default=DEFAULT_BALANCED_OUTPUT)
    parser.add_argument("--nearest-output", type=Path, default=DEFAULT_NEAREST_OUTPUT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    validated = load_validated_embeddings(args.embeddings)
    balanced = balanced_kmeans(
        validated.participant_ids,
        validated.vectors,
        n_groups=5,
        random_seed=args.seed,
    )
    nearest_five = nearest_neighbor_groups(
        validated.participant_ids,
        validated.vectors,
        group_size=5,
        n_groups=2,
        random_seed=args.seed,
    )
    nearest_ten = nearest_neighbor_groups(
        validated.participant_ids,
        validated.vectors,
        group_size=10,
        n_groups=2,
        random_seed=args.seed + 1,
    )
    balanced["embedding_metadata"] = validated.metadata
    nearest_five["embedding_metadata"] = validated.metadata
    nearest_ten["embedding_metadata"] = validated.metadata
    nearest = {
        "strategy": "nearest_neighbor",
        "configuration": {
            "base_random_seed": args.seed,
            "overlap_allowed": True,
            "experiments": [nearest_five["configuration"], nearest_ten["configuration"]],
        },
        "embedding_metadata": validated.metadata,
        "experiments": [
            {"group_size": 5, "groups": nearest_five["groups"]},
            {"group_size": 10, "groups": nearest_ten["groups"]},
        ],
    }
    save_json(args.balanced_output, balanced)
    save_json(args.nearest_output, nearest)
    print(f"Saved balanced groups to {args.balanced_output}")
    print(f"Saved nearest-neighbour groups to {args.nearest_output}")


if __name__ == "__main__":
    main()
