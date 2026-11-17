from __future__ import annotations

from pathlib import Path
import re
from typing import Callable, List, Optional, Sequence, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from PROSE.queries.fast_queries import FastAgent


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_PATH = (
    PROJECT_ROOT
    / "prose_abortion_dataset"
    / "abortion_generation_prose_cleaned.csv"
)
DEFAULT_SOURCE_PATH = (
    PROJECT_ROOT
    / "gsc_abortion"
    / "generative_social_choice"
    / "data"
    / "survey_data"
    / "abortion_generation_20240808_cleaned.csv"
)
VALIDATION_DATASET_PATH = (
    PROJECT_ROOT
    / "gsc_abortion"
    / "generative_social_choice"
    / "data"
    / "survey_data"
    / "abortion_validation_20240809_cleaned.csv"
)

RESPONSE_BLOCK_PATTERN = re.compile(
    r"Response:\n(.*?)(?=\n\nQuestion \(original step |\Z)",
    re.DOTALL,
)

AgentFactory = Callable[..., "FastAgent"]


def extract_response_blocks(data: str) -> List[str]:
    """Extract the preserved response text from a cleaned PROSE data field."""
    return RESPONSE_BLOCK_PATTERN.findall(data)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def validate_abortion_dataset(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    source_path: Optional[Path] = DEFAULT_SOURCE_PATH,
) -> pd.DataFrame:
    """Validate participant IDs, response blocks, and optional source preservation."""
    dataset = _read_csv(dataset_path)
    expected_columns = ["user_id", "data"]
    if list(dataset.columns) != expected_columns:
        raise ValueError(
            f"Expected columns {expected_columns}, found {list(dataset.columns)}"
        )
    if len(dataset) != 100:
        raise ValueError(f"Expected 100 participants, found {len(dataset)}")
    if dataset["user_id"].duplicated().any():
        raise ValueError("Duplicate participant IDs found")
    if dataset["user_id"].eq("").any():
        raise ValueError("Empty participant ID found")
    if dataset["data"].str.strip().eq("").any():
        raise ValueError("Empty participant data found")

    response_blocks = dataset["data"].map(extract_response_blocks)
    if not response_blocks.map(len).eq(5).all():
        counts = response_blocks.map(len).value_counts().to_dict()
        raise ValueError(f"Expected five response blocks per participant, found {counts}")

    if source_path is not None:
        source = _read_csv(source_path)
        source_longtext = source[
            source["question_type"] == "QuestionTypeIdentifier.LONGTEXT"
        ]
        source_responses = {
            participant_id: group.sort_values("step")["text"].tolist()
            for participant_id, group in source_longtext.groupby("user_id", sort=False)
        }
        for participant_id, blocks in zip(dataset["user_id"], response_blocks):
            if participant_id not in source_responses:
                raise ValueError(f"Participant {participant_id} is missing from source")
            if blocks != source_responses[participant_id]:
                raise ValueError(
                    f"Response text mismatch for participant {participant_id}"
                )

    return dataset


def load_abortion_agents(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    *,
    source_path: Optional[Path] = DEFAULT_SOURCE_PATH,
    model: str = "gpt-4o-mini-2024-07-18",
    prompt_type: str = "agreement_v1",
    specificity_prompt_type: str = "specificity_v1",
    temperature: int = 1,
    specificity_coeff: float = 0,
    agent_factory: Optional[AgentFactory] = None,
    participant_ids: Optional[Sequence[str]] = None,
) -> List[FastAgent]:
    """Load the cleaned abortion dataset as PROSE-compatible FastAgent objects."""
    if agent_factory is None:
        from PROSE.queries.fast_queries import FastAgent

        agent_factory = FastAgent

    dataset = validate_abortion_dataset(dataset_path, source_path)
    if participant_ids is not None:
        requested_ids = [str(participant_id) for participant_id in participant_ids]
        if len(requested_ids) != len(set(requested_ids)):
            raise ValueError("Duplicate participant IDs requested")
        available_ids = set(dataset["user_id"])
        missing_ids = sorted(set(requested_ids) - available_ids)
        if missing_ids:
            raise ValueError(f"Unknown participant IDs requested: {missing_ids}")
        dataset = dataset[dataset["user_id"].isin(requested_ids)].copy()
        dataset["_participant_order"] = dataset["user_id"].map(
            {participant_id: index for index, participant_id in enumerate(requested_ids)}
        )
        dataset = dataset.sort_values("_participant_order")

    agents = [
        agent_factory(
            id=str(row["user_id"]),
            data=row["data"],
            prompt_type=prompt_type,
            specificity_prompt_type=specificity_prompt_type,
            model=model,
            temperature=temperature,
            specificity_coeff=specificity_coeff,
            label="",
        )
        for _, row in dataset.iterrows()
    ]

    for agent in agents:
        agent.summary = agent.data

    expected_count = len(dataset)
    if len(agents) != expected_count or len(
        {agent.get_id() for agent in agents}
    ) != expected_count:
        raise ValueError(f"Adapter did not construct {expected_count} unique agents")
    return agents
