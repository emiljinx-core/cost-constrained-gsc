from abc import ABC, abstractmethod
from typing import Any, List, Protocol


class Agent(ABC):
    @abstractmethod
    def get_id(self) -> str:
        pass

    @abstractmethod
    def get_approval(self, statement: str) -> tuple[float, List[Any]]:
        pass

    @abstractmethod
    def get_description(self) -> str:
        """
        Return description of agent for use in generative queries (e.g. summary of user survey responses).
        """
        pass


class ApprovalScorer(Protocol):
    supports_specificity: bool

    def score(
        self,
        *,
        data: str,
        statement: str,
        prompt_type: str,
        system_prompt: str,
        prompt: str,
    ) -> tuple[float, List[Any]]:
        ...


class Generator(ABC):
    @abstractmethod
    def generate(self, agents: List[Agent]) -> tuple[list[str], List[Any]]:
        pass


class TargetedGenerator(ABC):
    @abstractmethod
    def get_name(self) -> str:
        # String representation of generator -- used for logging
        pass

    @abstractmethod
    def generate(
        self,
        agents: List[Agent],
        target_approval: float,
        target_budget: int,
    ) -> tuple[List[str], List[Any]]:
        pass
