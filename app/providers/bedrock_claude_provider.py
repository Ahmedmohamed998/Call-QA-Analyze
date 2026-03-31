import asyncio
import json
import logging
import time
from typing import Any

import boto3
from botocore.exceptions import ClientError
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


def _pydantic_to_tool_schema(model: type[BaseModel]) -> dict[str, Any]:
    """
    Convert a Pydantic model to a Bedrock tool input schema.
    Claude structured outputs work best via the tool-use pattern:
    we define the Pydantic schema as a tool, and Claude returns the
    structured data as a tool call.
    """
    schema = model.model_json_schema()

    def _clean_schema(s: dict) -> dict:
        s.pop("title", None)
        s.pop("default", None)
        s.pop("$defs", None)

        if "properties" in s:
            s["required"] = list(s["properties"].keys())
            for prop in s["properties"].values():
                _clean_schema(prop)

        if "items" in s and isinstance(s["items"], dict):
            _clean_schema(s["items"])

        # Resolve $ref references inline
        if "$ref" in s:
            s.pop("$ref")

        # Convert anyOf (Optional) to simpler representation
        if "anyOf" in s:
            non_null = [t for t in s["anyOf"] if t != {"type": "null"}]
            if len(non_null) == 1:
                merged = non_null[0]
                s.pop("anyOf")
                _clean_schema(merged)
                s.update(merged)

        return s

    # Resolve $defs references before cleaning
    defs = schema.pop("$defs", {})

    def _resolve_refs(obj: Any) -> Any:
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref_name = obj["$ref"].split("/")[-1]
                if ref_name in defs:
                    resolved = defs[ref_name].copy()
                    return _resolve_refs(resolved)
            return {k: _resolve_refs(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_resolve_refs(item) for item in obj]
        return obj

    schema = _resolve_refs(schema)
    return _clean_schema(schema)


class BedrockClaudeProvider(LLMProvider):
    """
    LLM provider using AWS Bedrock's Claude models.
    Uses the tool-use pattern to enforce structured outputs.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=settings.aws_bedrock_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
        )
        self.model_id = settings.aws_bedrock_inference_profile_id
        logger.info(
            "Bedrock Claude provider initialized | model=%s | region=%s",
            self.model_id,
            settings.aws_bedrock_region,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=16),
        retry=retry_if_exception_type((ClientError,)),
        before_sleep=lambda retry_state: logger.warning(
            "Retrying Bedrock call (attempt %d): %s",
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
        Call AWS Bedrock Claude with structured output via tool use.

        The Pydantic schema is defined as a tool, and Claude is instructed
        to call that tool with the analysis results as structured JSON.
        """
        start_time = time.perf_counter()
        tool_schema = _pydantic_to_tool_schema(response_model)

        tool_config = {
            "tools": [
                {
                    "toolSpec": {
                        "name": "submit_quality_analysis",
                        "description": (
                            "Submit the structured quality analysis results. "
                            "You MUST call this tool with your complete analysis."
                        ),
                        "inputSchema": {"json": tool_schema},
                    }
                }
            ],
            "toolChoice": {
                "tool": {"name": "submit_quality_analysis"}
            },
        }

        try:
            # Bedrock Converse API -- boto3 is synchronous, so we run it
            # in a thread pool to avoid blocking the async event loop.
            response = await asyncio.to_thread(
                self.client.converse,
                modelId=self.model_id,
                system=[{"text": system_prompt}],
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": user_prompt}],
                    }
                ],
                toolConfig=tool_config,
                inferenceConfig={
                    "temperature": 0.1,
                    "maxTokens": 4096,
                },
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Log usage
            usage = response.get("usage", {})
            logger.info(
                "Bedrock Claude call completed | latency=%.0fms | "
                "input_tokens=%d | output_tokens=%d",
                latency_ms,
                usage.get("inputTokens", 0),
                usage.get("outputTokens", 0),
            )

            # Extract tool use result from response
            output_message = response.get("output", {}).get("message", {})
            content_blocks = output_message.get("content", [])

            tool_result = None
            for block in content_blocks:
                if "toolUse" in block:
                    tool_result = block["toolUse"].get("input", {})
                    break

            if tool_result is None:
                raise LLMProviderError(
                    "Claude did not return a tool use response",
                    "bedrock_claude",
                )

            logger.debug("Raw tool result: %s", json.dumps(tool_result))

            # Parse and validate through Pydantic
            result = response_model.model_validate(tool_result)
            return result

        except ClientError as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "Bedrock Claude call failed | latency=%.0fms | error=%s",
                latency_ms,
                str(e),
            )
            raise
        except LLMProviderError:
            raise
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "Bedrock Claude processing failed | latency=%.0fms | error=%s",
                latency_ms,
                str(e),
            )
            raise LLMProviderError(
                f"Bedrock Claude call failed: {str(e)}", "bedrock_claude"
            ) from e

    async def close(self) -> None:
        """boto3 clients don't need explicit cleanup, but we log it."""
        logger.info("Bedrock Claude provider closed")
