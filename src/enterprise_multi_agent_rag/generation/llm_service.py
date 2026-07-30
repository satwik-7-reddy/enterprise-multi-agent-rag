"""Configured LLM generation with OpenAI and Bedrock Claude."""

import json
from typing import Any, Protocol

import boto3
from openai import OpenAI

from enterprise_multi_agent_rag.core.config import Settings


class LLMServiceError(Exception):
    """Raised when LLM configuration or generation fails."""


class InvalidPromptError(LLMServiceError):
    """Raised when a prompt contains no usable text."""


class LLMProvider(Protocol):
    """Minimal interface required by the LLM service."""

    def generate(self, prompt: str) -> str:
        """Generate answer text for an unchanged prompt."""


class OpenAILLMProvider:
    """Generate text with the OpenAI chat completions API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        client: Any | None = None,
    ) -> None:
        if client is None and not api_key:
            raise LLMServiceError(
                "OPENAI_API_KEY is required when the OpenAI LLM provider is selected."
            )
        self.model = model
        try:
            self._client = client or OpenAI(api_key=api_key)
        except Exception as exc:
            raise LLMServiceError(f"Could not create the OpenAI client: {exc}") from exc

    def generate(self, prompt: str) -> str:
        """Send the prompt as user content and return the first answer text."""
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            answer = response.choices[0].message.content
            if not isinstance(answer, str):
                raise LLMServiceError("OpenAI returned no text content.")
            return answer
        except LLMServiceError:
            raise
        except Exception as exc:
            raise LLMServiceError(
                f"OpenAI generation failed for model '{self.model}': {exc}"
            ) from exc


class BedrockClaudeProvider:
    """Generate text with Claude's native Bedrock messages request format."""

    def __init__(
        self,
        *,
        region: str = "us-east-1",
        model_id: str = "anthropic.claude-3-5-sonnet-20240620-v1:0",
        max_tokens: int = 1024,
        client: Any | None = None,
    ) -> None:
        self.model_id = model_id
        self.max_tokens = max_tokens
        try:
            self._client = client or boto3.client(
                "bedrock-runtime", region_name=region
            )
        except Exception as exc:
            raise LLMServiceError(
                f"Could not create the Bedrock Runtime client in region '{region}': {exc}"
            ) from exc

    def generate(self, prompt: str) -> str:
        """Invoke Claude and combine its returned text content blocks."""
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        try:
            response = self._client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(payload),
                contentType="application/json",
                accept="application/json",
            )
            body = response["body"]
            decoded = json.loads(body.read() if hasattr(body, "read") else body)
            text_blocks = [
                block["text"]
                for block in decoded["content"]
                if block.get("type") == "text" and isinstance(block.get("text"), str)
            ]
            if not text_blocks:
                raise LLMServiceError("Bedrock Claude returned no text content.")
            return "".join(text_blocks)
        except LLMServiceError:
            raise
        except Exception as exc:
            raise LLMServiceError(
                f"Bedrock Claude generation failed for model '{self.model_id}': {exc}"
            ) from exc


class LLMService:
    """Validate prompts and delegate generation to one configured provider."""

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    def generate(self, prompt: str) -> str:
        """Return plain generated text for an unchanged prompt."""
        if not prompt or not prompt.strip():
            raise InvalidPromptError("Prompt must not be empty or whitespace-only.")
        return self.provider.generate(prompt)


def create_llm_service(settings: Settings) -> LLMService:
    """Create an LLM service from case-insensitive application configuration."""
    provider_name = settings.llm_provider.strip().lower()
    if provider_name == "openai":
        provider: LLMProvider = OpenAILLMProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_chat_model,
        )
    elif provider_name == "bedrock":
        provider = BedrockClaudeProvider(
            region=settings.aws_region,
            model_id=settings.bedrock_chat_model_id,
            max_tokens=settings.llm_max_tokens,
        )
    else:
        raise LLMServiceError(f"Unsupported LLM provider: '{settings.llm_provider}'.")
    return LLMService(provider)
