"""Environment-backed application configuration."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "enterprise-multi-agent-rag"
    app_env: Literal["development", "test", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    llm_provider: str = "openai"
    openai_chat_model: str = "gpt-4o-mini"
    bedrock_chat_model_id: str = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    llm_max_tokens: int = 1024
    embedding_provider: str = "bedrock"
    aws_region: str = "us-east-1"
    bedrock_embedding_model: str = "amazon.titan-embed-text-v2:0"
    bedrock_embedding_dimensions: int = 1024
    bedrock_embedding_normalize: bool = True
    openai_api_key: str | None = Field(default=None, repr=False)
    openai_embedding_model: str = "text-embedding-3-small"
    anthropic_api_key: str | None = Field(default=None, repr=False)


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings()
