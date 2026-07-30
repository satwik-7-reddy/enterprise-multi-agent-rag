"""Unit tests for configured LLM text generation."""

import io
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from enterprise_multi_agent_rag.core.config import Settings
from enterprise_multi_agent_rag.generation.llm_service import (
    BedrockClaudeProvider,
    InvalidPromptError,
    LLMService,
    LLMServiceError,
    OpenAILLMProvider,
    create_llm_service,
)


class FakeLLMProvider:
    """Record prompts and return deterministic answer text."""

    def __init__(self, answer: str = "Generated answer.") -> None:
        self.answer = answer
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer


def test_service_generates_plain_text_and_preserves_prompt() -> None:
    provider = FakeLLMProvider("Employees receive 15 vacation days.")
    service = LLMService(provider)
    prompt = "  Final prompt with intentional spacing.\n"

    answer = service.generate(prompt)

    assert answer == "Employees receive 15 vacation days."
    assert isinstance(answer, str)
    assert provider.prompts == [prompt]


@pytest.mark.parametrize("prompt", ["", " \n\t "])
def test_service_rejects_empty_or_whitespace_prompt(prompt: str) -> None:
    provider = FakeLLMProvider()

    with pytest.raises(InvalidPromptError, match="must not be empty"):
        LLMService(provider).generate(prompt)

    assert provider.prompts == []


def test_openai_provider_sends_user_message_and_returns_first_text() -> None:
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content="OpenAI answer.")),
            SimpleNamespace(message=SimpleNamespace(content="Unused answer.")),
        ]
    )
    provider = OpenAILLMProvider(client=client, model="gpt-test")

    answer = provider.generate("Exact final prompt")

    assert answer == "OpenAI answer."
    client.chat.completions.create.assert_called_once_with(
        model="gpt-test",
        messages=[{"role": "user", "content": "Exact final prompt"}],
    )


def test_openai_provider_wraps_errors() -> None:
    client = Mock()
    client.chat.completions.create.side_effect = RuntimeError("offline")

    with pytest.raises(LLMServiceError, match="OpenAI generation failed"):
        OpenAILLMProvider(client=client).generate("prompt")


def test_openai_provider_requires_key_without_injected_client() -> None:
    with pytest.raises(LLMServiceError, match="OPENAI_API_KEY"):
        OpenAILLMProvider()


def test_bedrock_provider_uses_claude_messages_format() -> None:
    client = Mock()
    client.invoke_model.return_value = {
        "body": io.BytesIO(
            json.dumps(
                {
                    "content": [
                        {"type": "text", "text": "Bedrock "},
                        {"type": "text", "text": "answer."},
                    ]
                }
            ).encode()
        )
    }
    provider = BedrockClaudeProvider(
        client=client,
        model_id="anthropic.claude-test",
        max_tokens=321,
    )

    answer = provider.generate("Exact final prompt")

    assert answer == "Bedrock answer."
    request = client.invoke_model.call_args.kwargs
    assert request["modelId"] == "anthropic.claude-test"
    assert request["contentType"] == "application/json"
    assert request["accept"] == "application/json"
    assert json.loads(request["body"]) == {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 321,
        "messages": [{"role": "user", "content": "Exact final prompt"}],
    }


def test_bedrock_provider_wraps_errors() -> None:
    client = Mock()
    client.invoke_model.side_effect = RuntimeError("offline")

    with pytest.raises(LLMServiceError, match="Bedrock Claude generation failed"):
        BedrockClaudeProvider(client=client).generate("prompt")


def test_factory_selects_openai_case_insensitively() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="OpEnAI",
        openai_api_key="test-key",
        openai_chat_model="gpt-test",
    )
    with patch(
        "enterprise_multi_agent_rag.generation.llm_service.OpenAILLMProvider"
    ) as provider_class:
        service = create_llm_service(settings)

    assert service.provider is provider_class.return_value
    provider_class.assert_called_once_with(api_key="test-key", model="gpt-test")


def test_factory_selects_bedrock() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="bedrock",
        aws_region="us-west-2",
        bedrock_chat_model_id="anthropic.claude-test",
        llm_max_tokens=456,
    )
    with patch(
        "enterprise_multi_agent_rag.generation.llm_service.BedrockClaudeProvider"
    ) as provider_class:
        service = create_llm_service(settings)

    assert service.provider is provider_class.return_value
    provider_class.assert_called_once_with(
        region="us-west-2",
        model_id="anthropic.claude-test",
        max_tokens=456,
    )


def test_factory_rejects_unsupported_provider() -> None:
    settings = Settings(_env_file=None, llm_provider="unknown")

    with pytest.raises(LLMServiceError, match="Unsupported LLM provider"):
        create_llm_service(settings)
