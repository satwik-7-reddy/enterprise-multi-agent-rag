"""Unit tests for provider-independent embedding generation."""

import io
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from enterprise_multi_agent_rag.chunking.models import DocumentChunk
from enterprise_multi_agent_rag.core.config import Settings
from enterprise_multi_agent_rag.embeddings.exceptions import (
    EmbeddingConfigurationError,
    EmbeddingDimensionMismatchError,
    EmbeddingProviderError,
    EmptyEmbeddingInputError,
    UnsupportedEmbeddingProviderError,
)
from enterprise_multi_agent_rag.embeddings.factory import create_embedding_provider
from enterprise_multi_agent_rag.embeddings.providers import (
    BaseEmbeddingProvider,
    BedrockEmbeddingProvider,
    OpenAIEmbeddingProvider,
)
from enterprise_multi_agent_rag.embeddings.service import EmbeddingService


class FakeProvider(BaseEmbeddingProvider):
    """Deterministic provider used to isolate service behavior."""

    def __init__(self, vectors: list[list[object]]) -> None:
        self.vectors = vectors
        self.received: list[str] | None = None

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-model"

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]  # type: ignore[return-value]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.received = texts
        return self.vectors  # type: ignore[return-value]


def _chunk(index: int, content: str = "generated content") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"chunk-{index}",
        document_id="document-1",
        chunk_index=index,
        content=content,
        start_character=0,
        end_character=len(content),
        metadata={"source": "generated", "nested": {"value": index}},
    )


def test_bedrock_request_payload_and_response() -> None:
    client = Mock()
    vector = [float(index) for index in range(256)]
    client.invoke_model.return_value = {
        "body": io.BytesIO(json.dumps({"embedding": vector}).encode())
    }
    provider = BedrockEmbeddingProvider(
        client=client,
        region="us-west-2",
        model_id="amazon.titan-embed-text-v2:0",
        dimensions=256,
        normalize=False,
    )

    result = provider.embed_text("hello")

    assert result == vector
    request = client.invoke_model.call_args.kwargs
    assert request["modelId"] == "amazon.titan-embed-text-v2:0"
    assert request["contentType"] == "application/json"
    assert request["accept"] == "application/json"
    assert json.loads(request["body"]) == {
        "inputText": "hello",
        "dimensions": 256,
        "normalize": False,
    }


def test_bedrock_creates_runtime_client_in_configured_region() -> None:
    with patch(
        "enterprise_multi_agent_rag.embeddings.providers.boto3.client"
    ) as client_factory:
        BedrockEmbeddingProvider(region="eu-west-1")

    client_factory.assert_called_once_with("bedrock-runtime", region_name="eu-west-1")


def test_bedrock_batch_repeats_native_calls_in_order() -> None:
    client = Mock()
    client.invoke_model.side_effect = [
        {"body": io.BytesIO(json.dumps({"embedding": [1.0] * 256}).encode())},
        {"body": io.BytesIO(json.dumps({"embedding": [2.0] * 256}).encode())},
    ]
    provider = BedrockEmbeddingProvider(client=client, dimensions=256)

    vectors = provider.embed_texts(["first", "second"])

    assert vectors == [[1.0] * 256, [2.0] * 256]
    assert [
        json.loads(call.kwargs["body"])["inputText"]
        for call in client.invoke_model.call_args_list
    ] == ["first", "second"]


@pytest.mark.parametrize("dimensions", [0, 255, 768, 2048])
def test_bedrock_rejects_invalid_dimensions(dimensions: int) -> None:
    with pytest.raises(EmbeddingConfigurationError, match="256, 512, or 1024"):
        BedrockEmbeddingProvider(client=Mock(), dimensions=dimensions)


def test_bedrock_translates_client_and_response_failures() -> None:
    failing_client = Mock()
    failing_client.invoke_model.side_effect = RuntimeError("offline")
    provider = BedrockEmbeddingProvider(client=failing_client, dimensions=256)
    with pytest.raises(EmbeddingProviderError, match="Bedrock embedding request failed"):
        provider.embed_text("text")

    malformed_client = Mock()
    malformed_client.invoke_model.return_value = {
        "body": io.BytesIO(json.dumps({"embedding": [1.0]}).encode())
    }
    provider = BedrockEmbeddingProvider(client=malformed_client, dimensions=256)
    with pytest.raises(EmbeddingProviderError, match="expected 256"):
        provider.embed_text("text")


def test_openai_uses_batch_request_and_response_indexes() -> None:
    client = Mock()
    client.embeddings.create.return_value = SimpleNamespace(
        data=[
            SimpleNamespace(index=1, embedding=[3, 4]),
            SimpleNamespace(index=0, embedding=[1, 2]),
        ]
    )
    provider = OpenAIEmbeddingProvider(client=client, model="text-embedding-test")

    vectors = provider.embed_texts(["first", "second"])

    assert vectors == [[1.0, 2.0], [3.0, 4.0]]
    client.embeddings.create.assert_called_once_with(
        model="text-embedding-test", input=["first", "second"]
    )


def test_openai_translates_sdk_errors() -> None:
    client = Mock()
    client.embeddings.create.side_effect = RuntimeError("offline")
    with pytest.raises(EmbeddingProviderError, match="OpenAI embedding request failed"):
        OpenAIEmbeddingProvider(client=client).embed_texts(["text"])


def test_openai_requires_key_without_injected_client() -> None:
    with pytest.raises(EmbeddingConfigurationError, match="OPENAI_API_KEY"):
        OpenAIEmbeddingProvider()


def test_factory_selects_bedrock_case_insensitively() -> None:
    settings = Settings(
        _env_file=None,
        embedding_provider="BeDrOcK",
        aws_region="ap-south-1",
        bedrock_embedding_dimensions=512,
    )
    with patch(
        "enterprise_multi_agent_rag.embeddings.factory.BedrockEmbeddingProvider"
    ) as provider_class:
        provider = create_embedding_provider(settings)

    assert provider is provider_class.return_value
    provider_class.assert_called_once_with(
        region="ap-south-1",
        model_id="amazon.titan-embed-text-v2:0",
        dimensions=512,
        normalize=True,
    )


def test_factory_selects_openai() -> None:
    settings = Settings(
        _env_file=None,
        embedding_provider="openai",
        openai_api_key="test-key",
        openai_embedding_model="test-model",
    )
    with patch(
        "enterprise_multi_agent_rag.embeddings.factory.OpenAIEmbeddingProvider"
    ) as provider_class:
        create_embedding_provider(settings)

    provider_class.assert_called_once_with(api_key="test-key", model="test-model")


def test_factory_rejects_unsupported_provider() -> None:
    settings = Settings(_env_file=None, embedding_provider="unknown")
    with pytest.raises(UnsupportedEmbeddingProviderError, match="unknown"):
        create_embedding_provider(settings)


def test_service_preserves_order_and_is_deterministic() -> None:
    chunks = [_chunk(0, "first"), _chunk(1, "second")]
    provider = FakeProvider([[1, 2], [3, 4]])
    service = EmbeddingService(provider)

    first = service.embed_chunks(chunks)
    second = service.embed_chunks(chunks)

    assert provider.received == ["first", "second"]
    assert [item.chunk_id for item in first] == ["chunk-0", "chunk-1"]
    assert [item.embedding for item in first] == [[1.0, 2.0], [3.0, 4.0]]
    assert first == second


def test_service_copies_metadata_without_mutation() -> None:
    chunk = _chunk(0)
    original_metadata = chunk.metadata.copy()

    embedded = EmbeddingService(FakeProvider([[1, 2]])).embed_chunks([chunk])[0]
    embedded.metadata["added"] = True
    embedded.metadata["nested"]["value"] = 99

    assert chunk.metadata == original_metadata
    assert "added" not in chunk.metadata


def test_service_treats_empty_list_as_no_op() -> None:
    provider = FakeProvider([])
    assert EmbeddingService(provider).embed_chunks([]) == []
    assert provider.received is None


def test_service_rejects_whitespace_only_chunk() -> None:
    with pytest.raises(EmptyEmbeddingInputError, match="no embeddable text"):
        EmbeddingService(FakeProvider([])).embed_chunks([_chunk(0, " \n\t")])


@pytest.mark.parametrize("vectors", [[], [[1, 2]], [[1, 2], [3, 4], [5, 6]]])
def test_service_rejects_wrong_result_count(vectors: list[list[object]]) -> None:
    with pytest.raises(EmbeddingProviderError, match="vectors for 2 chunks"):
        EmbeddingService(FakeProvider(vectors)).embed_chunks([_chunk(0), _chunk(1)])


def test_service_rejects_empty_vector() -> None:
    with pytest.raises(EmptyEmbeddingInputError, match="empty vector"):
        EmbeddingService(FakeProvider([[]])).embed_chunks([_chunk(0)])


@pytest.mark.parametrize("value", ["invalid", None, True])
def test_service_rejects_non_numeric_values(value: object) -> None:
    with pytest.raises(EmbeddingProviderError, match="non-numeric"):
        EmbeddingService(FakeProvider([[1.0, value]])).embed_chunks([_chunk(0)])


def test_service_rejects_inconsistent_dimensions() -> None:
    with pytest.raises(EmbeddingDimensionMismatchError, match="inconsistent"):
        EmbeddingService(FakeProvider([[1, 2], [3, 4, 5]])).embed_chunks(
            [_chunk(0), _chunk(1)]
        )
