"""Embedding provider abstractions and SDK-specific implementations."""

import json
from abc import ABC, abstractmethod
from typing import Any

import boto3
from openai import OpenAI

from enterprise_multi_agent_rag.embeddings.exceptions import (
    EmbeddingConfigurationError,
    EmbeddingProviderError,
)


class BaseEmbeddingProvider(ABC):
    """Provider-independent embedding interface."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the stable provider identifier."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the provider model identifier."""

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Embed one text input."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed text inputs while preserving their order."""


class BedrockEmbeddingProvider(BaseEmbeddingProvider):
    """Amazon Titan Text Embeddings V2 provider through Bedrock Runtime."""

    VALID_DIMENSIONS = frozenset({256, 512, 1024})

    def __init__(
        self,
        *,
        region: str = "us-east-1",
        model_id: str = "amazon.titan-embed-text-v2:0",
        dimensions: int = 1024,
        normalize: bool = True,
        client: Any | None = None,
    ) -> None:
        if dimensions not in self.VALID_DIMENSIONS:
            raise EmbeddingConfigurationError(
                "Titan V2 dimensions must be one of 256, 512, or 1024."
            )
        self.region = region
        self.model_id = model_id
        self.dimensions = dimensions
        self.normalize = normalize
        try:
            self._client = client or boto3.client("bedrock-runtime", region_name=region)
        except Exception as exc:
            raise EmbeddingProviderError(
                f"Could not create the Bedrock Runtime client in region '{region}': {exc}"
            ) from exc

    @property
    def provider_name(self) -> str:
        return "bedrock"

    @property
    def model_name(self) -> str:
        return self.model_id

    def embed_text(self, text: str) -> list[float]:
        payload = {
            "inputText": text,
            "dimensions": self.dimensions,
            "normalize": self.normalize,
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
            embedding = decoded["embedding"]
        except Exception as exc:
            raise EmbeddingProviderError(
                f"Bedrock embedding request failed for model '{self.model_id}': {exc}"
            ) from exc

        if (
            not isinstance(embedding, list)
            or len(embedding) != self.dimensions
            or any(
                isinstance(value, bool) or not isinstance(value, int | float)
                for value in embedding
            )
        ):
            raise EmbeddingProviderError(
                "Bedrock returned an invalid embedding vector "
                f"(expected {self.dimensions} numeric values)."
            )
        return [float(value) for value in embedding]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI native batch embedding provider."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        client: Any | None = None,
    ) -> None:
        if client is None and not api_key:
            raise EmbeddingConfigurationError(
                "OPENAI_API_KEY is required when the OpenAI embedding provider is selected."
            )
        self.model = model
        try:
            self._client = client or OpenAI(api_key=api_key)
        except Exception as exc:
            raise EmbeddingProviderError(f"Could not create the OpenAI client: {exc}") from exc

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self.model

    def embed_text(self, text: str) -> list[float]:
        vectors = self.embed_texts([text])
        if len(vectors) != 1:
            raise EmbeddingProviderError("OpenAI did not return exactly one embedding.")
        return vectors[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self._client.embeddings.create(model=self.model, input=texts)
            ordered = sorted(response.data, key=lambda item: item.index)
            return [[float(value) for value in item.embedding] for item in ordered]
        except Exception as exc:
            raise EmbeddingProviderError(
                f"OpenAI embedding request failed for model '{self.model}': {exc}"
            ) from exc
