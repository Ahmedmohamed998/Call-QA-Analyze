from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class LLMProvider(ABC):
    """
    Abstract interface for LLM providers.

    All providers must implement the `analyze` method which takes
    a system prompt, user prompt, and a Pydantic response model type,
    and returns a validated instance of that model.
    """

    @abstractmethod
    async def analyze(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
    ) -> BaseModel:
        """
        Send prompts to the LLM and return a structured response.

        Args:
            system_prompt: The system-level instructions for the LLM.
            user_prompt: The user-level content (transcript data).
            response_model: Pydantic model class to validate and structure output.

        Returns:
            An instance of response_model populated with the LLM's analysis.

        Raises:
            LLMProviderError: If the LLM call fails after retries.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Clean up any resources held by the provider."""
        ...


class LLMProviderError(Exception):
    """Raised when an LLM provider fails to produce a valid response."""

    def __init__(self, message: str, provider: str, details: Any = None):
        self.provider = provider
        self.details = details
        super().__init__(f"[{provider}] {message}")
