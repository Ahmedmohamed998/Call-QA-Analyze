import json
import logging
import time

from openai import AsyncAzureOpenAI, APIError, APITimeoutError, RateLimitError
from pydantic import BaseModel
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from app.config import Settings
from app.providers.base import LLMProvider, LLMProviderError

logger = logging.getLogger(__name__)


class AzureOpenAIProvider(LLMProvider):
    """
    LLM provider using Azure OpenAI's chat completions API.
    Uses the SDK's built-in beta.chat.completions.parse() for
    native Pydantic structured output support.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = AsyncAzureOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
        )
        self.deployment = settings.azure_openai_chat_deployment
        self.max_retries = settings.max_retries
        logger.info(
            "Azure OpenAI provider initialized | deployment=%s | endpoint=%s",
            self.deployment,
            settings.azure_openai_endpoint,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=16),
        retry=retry_if_exception_type((APIError, APITimeoutError, RateLimitError)),
        before_sleep=lambda retry_state: logger.warning(
            "Retrying Azure OpenAI call (attempt %d): %s",
            retry_state.attempt_number,
            str(retry_state.outcome.exception()) if retry_state.outcome else "unknown",
        ),
    )
    async def analyze(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
    ) -> BaseModel:
        """
        Call Azure OpenAI with structured output enforcement.

        Uses the SDK's beta.chat.completions.parse() method which
        natively handles Pydantic models as response_format, including
        all schema conversion, $ref resolution, and validation.
        """
        start_time = time.perf_counter()

        try:
            response = await self.client.beta.chat.completions.parse(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=response_model,
                max_completion_tokens=4096,
                timeout=self.settings.request_timeout,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Log observability data
            usage = response.usage
            logger.info(
                "Azure OpenAI call completed | latency=%.0fms | "
                "prompt_tokens=%d | completion_tokens=%d | total_tokens=%d",
                latency_ms,
                usage.prompt_tokens if usage else 0,
                usage.completion_tokens if usage else 0,
                usage.total_tokens if usage else 0,
            )

            # Extract parsed Pydantic model from response
            message = response.choices[0].message

            if message.refusal:
                raise LLMProviderError(
                    f"Model refused to respond: {message.refusal}",
                    "azure_openai",
                )

            if message.parsed is None:
                # Fallback: try parsing raw content
                logger.warning("Parsed response is None, attempting manual parse")
                content = message.content
                logger.debug("Raw LLM response: %s", content)
                parsed = json.loads(content)
                result = response_model.model_validate(parsed)
                return result

            logger.debug("Parsed response: %s", message.parsed)
            return message.parsed

        except json.JSONDecodeError as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "Failed to parse LLM JSON response | latency=%.0fms | error=%s",
                latency_ms,
                str(e),
            )
            raise LLMProviderError(
                "LLM returned invalid JSON", "azure_openai", str(e)
            ) from e
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "Azure OpenAI call failed | latency=%.0fms | error=%s",
                latency_ms,
                str(e),
            )
            if isinstance(e, LLMProviderError):
                raise
            raise LLMProviderError(
                f"Azure OpenAI call failed: {str(e)}", "azure_openai"
            ) from e

    async def close(self) -> None:
        """Close the async client."""
        await self.client.close()
        logger.info("Azure OpenAI provider closed")
