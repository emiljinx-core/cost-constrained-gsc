from abc import ABC
from PROSE.queries.query_interface import Agent
from PROSE.queries.prompts.summarize_prompts import (
    get_prompt_template_from_prompt_type,
    get_system_prompt_from_prompt_type,
)
import pandas as pd
from typing import Any


class SummaryAgent(Agent, ABC):
    """
    General implementation of basic summarization, used downstream by FewshotAgent, CoTAgent, PureFewshotAgent
    """

    def generate_summary(
        self, prompt_type: str = "v1", overwrite: bool = False
    ) -> tuple[str, list[Any]]:
        # Copied from CoTAgent
        if self.summary and not overwrite:
            raise ValueError(
                "Summary already exists. Set overwrite=True to regenerate."
            )
        if isinstance(self.data, pd.DataFrame):
            user_data = self.data.to_json(orient="records")
        elif isinstance(self.data, str):
            user_data = self.data
        else:
            raise NotImplementedError
        prompt = get_prompt_template_from_prompt_type(prompt_type=prompt_type).format(
            user_data=user_data
        )
        messages = [
            {
                "role": "system",
                "content": get_system_prompt_from_prompt_type(prompt_type=prompt_type),
            },
            {"role": "user", "content": prompt},
        ]
        if self.summary_llm is None:
            raise RuntimeError(
                "No summary provider is configured for this agent."
            )
        # validator = get_validator_from_prompt_type(prompt_type=prompt_type)
        # TODO: integrate validator by switching to new JSON mode
        summary, _, log = self.summary_llm.call(
            messages=messages, response_format={"type": "json_object"}
        )
        self.summary = summary
        return summary, [log]

    def get_description(self) -> str:
        """
        Return LLM-generated summary of agent. For use in generative queries
        """
        assert self.summary, "Summary not yet generated. Call generate_summary() first."
        return self.summary
