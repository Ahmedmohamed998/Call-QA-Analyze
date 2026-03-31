from enum import Enum
from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings


class LLMProviderType(str, Enum):
    """Supported LLM provider backends."""

    AZURE_OPENAI = "azure_openai"
    BEDROCK_CLAUDE = "bedrock_claude"


class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    # --- Azure OpenAI ---
    azure_openai_api_key: Optional[str] = None
    azure_openai_endpoint: Optional[str] = None
    azure_openai_api_version: str = "2024-02-15-preview"
    azure_openai_chat_deployment: str = "gpt-5.3-chat"

    # --- AWS Bedrock (Claude) ---
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_bedrock_region: str = "us-east-1"
    aws_bedrock_inference_profile_id: str = (
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    )

    # --- General ---
    llm_provider: LLMProviderType = LLMProviderType.AZURE_OPENAI
    log_level: str = "INFO"
    max_retries: int = 3
    request_timeout: int = 60

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }

    @field_validator("llm_provider", mode="before")
    @classmethod
    def normalize_provider(cls, v: str) -> str:
        if isinstance(v, str):
            return v.lower().strip()
        return v

    def validate_provider_credentials(self) -> None:
        """Ensure the selected provider has all required credentials."""
        if self.llm_provider == LLMProviderType.AZURE_OPENAI:
            if not self.azure_openai_api_key or not self.azure_openai_endpoint:
                raise ValueError(
                    "Azure OpenAI requires AZURE_OPENAI_API_KEY and "
                    "AZURE_OPENAI_ENDPOINT to be set."
                )
        elif self.llm_provider == LLMProviderType.BEDROCK_CLAUDE:
            if not self.aws_access_key_id or not self.aws_secret_access_key:
                raise ValueError(
                    "AWS Bedrock requires AWS_ACCESS_KEY_ID and "
                    "AWS_SECRET_ACCESS_KEY to be set."
                )


def get_settings() -> Settings:
    """Factory function to create and validate settings."""
    settings = Settings()
    settings.validate_provider_credentials()
    return settings
