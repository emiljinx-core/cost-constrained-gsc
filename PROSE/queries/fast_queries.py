from typing import Any, Dict, List, Optional, Union, Tuple
from PROSE.queries.query_interface import Agent, TargetedGenerator
from PROSE.queries.summary_agent import SummaryAgent
from PROSE.queries.approval_scorers import OpenAIApprovalScorer
from PROSE.queries.query_interface import ApprovalScorer
from PROSE.utils.ollama_provider import ChatProvider
from PROSE.queries.prompts.fast_disc_query_prompts import (
    get_prompt_disc,
    get_system_prompt_disc,
    get_llm_approval_levels,
)
from PROSE.queries.prompts.fast_gen_query_prompts import (
    get_system_prompt_fast_generator,
)
from PROSE.queries.prompts.fast_gen_no_budget_query_prompts import (
    get_system_prompt_fast_no_budget_generator,
)
from PROSE.utils.helper_functions import count_words
import random
import re


DEFAULT_MODEL = "gpt-4o-2024-11-20"


class GenerationFormatError(ValueError):
    """Raised when a generator response has no usable opinion."""


def process_llm_approval_levels(
    llm_approval_levels: Union[Dict[str, int], List[int]],
) -> Tuple[int, int, int, str]:
    if isinstance(llm_approval_levels, dict):
        max_approval_level = max(llm_approval_levels.keys())
        min_approval_level = min(llm_approval_levels.keys())
        num_approval_levels = len(llm_approval_levels)
        str_approval_levels = str(llm_approval_levels)
    else:
        max_approval_level = max(llm_approval_levels)
        min_approval_level = min(llm_approval_levels)
        num_approval_levels = len(llm_approval_levels)
        str_approval_levels = str(llm_approval_levels)
    assert min_approval_level < max_approval_level
    assert num_approval_levels > 1
    return (
        max_approval_level,
        min_approval_level,
        num_approval_levels,
        str_approval_levels,
    )


class FastAgent(SummaryAgent):
    def __init__(
        self,
        *,
        id: str,
        data: str,
        prompt_type: str,
        specificity_prompt_type: str,
        model: str = "gpt-4o-mini-2024-07-18",
        # model: str = DEFAULT_MODEL,
        temperature: int = 1,
        specificity_coeff: float = 0,
        label: str = " ",
        approval_scorer: Optional[ApprovalScorer] = None,
        summary_provider: Optional[ChatProvider] = None,
    ):
        self.embedding = None
        self.id = id
        self.data = data
        self.llm_approval_levels = get_llm_approval_levels(prompt_type)
        self.prompt_type = prompt_type
        self.specificity_prompt_type = specificity_prompt_type
        self.label = label
        self.approval_scorer = approval_scorer or OpenAIApprovalScorer(
            model=model, temperature=temperature
        )
        self.llm = getattr(self.approval_scorer, "llm", None)
        self.summary = None
        if summary_provider is not None:
            self.summary_llm = summary_provider
        elif approval_scorer is None:
            from PROSE.utils.llm_tools import GPT

            self.summary_llm = GPT(model=model, temperature=0)
        else:
            self.summary_llm = None
        (
            self.max_approval_level,
            self.min_approval_level,
            self.num_approval_levels,
            self.str_approval_levels,
        ) = process_llm_approval_levels(self.llm_approval_levels)

        # Process specificity levels
        self.llm_specificity_levels = get_llm_approval_levels(specificity_prompt_type)
        self.specificity_coeff = specificity_coeff
        self.specificity_enabled = getattr(
            self.approval_scorer, "supports_specificity", False
        )
        if not self.specificity_enabled and specificity_coeff != 0:
            raise ValueError(
                "specificity_coeff must be zero when the injected approval "
                "scorer does not support specificity. Ollama agreement-only "
                "scoring requires specificity_coeff=0."
            )
        self.max_specificity_level, self.min_specificity_level, _, _ = (
            process_llm_approval_levels(self.llm_specificity_levels)
        )

    def get_id(self) -> str:
        return self.id

    def get_label(self) -> str:
        return self.label

    def get_data(self):
        return self.data

    def get_embedding(self):
        return self.embedding

    def set_embedding(self, embedding):
        self.embedding = embedding

    def get_approval(self, statement: str) -> tuple[float, List[Any]]:
        if not self.specificity_enabled:
            return self._make_query(statement, self.prompt_type)

        specificity, log_list_specificity = self._make_query(
            statement, self.specificity_prompt_type
        )
        approval, log_list_approval = self._make_query(statement, self.prompt_type)
        log_list = log_list_specificity + log_list_approval

        specificity_deduction = (
            self.specificity_coeff
            * (self.max_specificity_level - specificity)
            / (self.max_specificity_level - self.min_specificity_level)
        )

        score = approval - specificity_deduction

        return score, log_list

    def _make_query(
        self, statement: str, prompt_type: str
    ) -> tuple[float, List[Any]]:
        system_prompt = get_system_prompt_disc(prompt_type).format(
            min_approval_level=self.min_approval_level,
            max_approval_level=self.max_approval_level,
        )
        prompt = get_prompt_disc(prompt_type).format(
            data=self.data,
            statement=statement,
            str_approval_levels=self.str_approval_levels,
            min_approval_level=self.min_approval_level,
            max_approval_level=self.max_approval_level,
        )
        return self.approval_scorer.score(
            data=self.data,
            statement=statement,
            prompt_type=prompt_type,
            system_prompt=system_prompt,
            prompt=prompt,
        )


class FastGenerator(TargetedGenerator):
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        prompt_type: str = "v1",
        temperature: float = 1,
        openai_seed: int = 0,
        target_approval: Optional[float] = None,
        target_budget: Optional[int] = None,
        text_provider: Optional[ChatProvider] = None,
    ):
        if text_provider is not None:
            self.llm = text_provider
        else:
            from PROSE.utils.llm_tools import GPT

            self.llm = GPT(
                temperature=temperature, model=model, openai_seed=openai_seed
            )
        self.prompt_type = prompt_type
        self.target_approval = target_approval
        self.target_budget = target_budget

    def get_name(self) -> str:
        return (
            self.__class__.__name__
            + f"__approval_{self.target_approval}__budget_{self.target_budget}"
        )

    def generate(
        self,
        agents: List[Agent],
        target_approval: float,
        target_budget: int,
    ) -> tuple[list[str], List[Any]]:
        if target_approval is None:
            target_approval = self.target_approval
        if target_budget is None:
            target_budget = self.target_budget
        assert target_budget is not None
        # assert target_approval is not None
        if target_budget > -1:
            system_prompt = get_system_prompt_fast_generator(self.prompt_type).format(
                target_budget=target_budget, target_approval=target_approval
            )
        else:
            system_prompt = get_system_prompt_fast_no_budget_generator(
                self.prompt_type
            ).format(target_approval=target_approval)

        random.shuffle(agents)
        agents_descriptions_str = "\n\n".join(
            [f"User {agent.get_id()}: {agent.get_description()}" for agent in agents]
        )
        prompt = (
            f"User information:\n{agents_descriptions_str}\n\nNow write the opinion."
        )

        response, _, log = self.llm.call(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        )

        opinion_match = re.fullmatch(
            r"\s*<opinion>(.*?)</opinion>\s*", response, flags=re.DOTALL
        )
        if opinion_match is None:
            raise GenerationFormatError(
                "Generator response must contain exactly one <opinion>...</opinion> block."
            )
        statement = opinion_match.group(1)
        if "<opinion>" in statement or "</opinion>" in statement:
            raise GenerationFormatError(
                "Generator response must contain exactly one <opinion>...</opinion> block."
            )
        if not statement.strip():
            raise GenerationFormatError(
                "Generator response contained an empty <opinion>."
            )
        provider_config = getattr(self.llm, "config", None)
        if not (
            getattr(provider_config, "log_responses", None) is False
        ):
            log["statement"] = statement

        return [statement], [log]


class StrictFastGenerator(FastGenerator):
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        prompt_type: str = "v1",
        temperature: float = 1,
        openai_seed: int = 0,
        target_approval: Optional[float] = None,
        target_budget: Optional[int] = None,
        num_retries: int = 5,
        text_provider: Optional[Any] = None,
    ):
        """
        Same as FastGenerator, but retry up to num_retries times if the statement generated is out of budget.
        If it's still out of budget after, then print a warning.
        """
        super().__init__(
            model=model,
            prompt_type=prompt_type,
            temperature=temperature,
            openai_seed=openai_seed,
            target_approval=target_approval,
            target_budget=target_budget,
            text_provider=text_provider,
        )
        self.num_retries = num_retries

    def generate(
        self,
        agents: List[Agent],
        target_approval: float,
        target_budget: int,
    ) -> tuple[list[str], List[Any]]:
        for _ in range(self.num_retries):
            statement_list, log_list = super().generate(
                agents=agents,
                target_approval=target_approval,
                target_budget=target_budget,
            )

            assert len(statement_list) == 1

            word_count = count_words(statement_list[0])
            if word_count <= target_budget:
                return statement_list, log_list

        print(
            f"Warning: after {self.num_retries} attempts, could not generate statement within word budget of {target_budget}. (Generated statement instead has {word_count} words.)"
        )

        return statement_list, log_list
