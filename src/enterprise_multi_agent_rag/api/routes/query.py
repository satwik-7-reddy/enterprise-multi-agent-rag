"""Question-answering API route."""

from functools import lru_cache

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, field_validator

from enterprise_multi_agent_rag.core.config import get_settings
from enterprise_multi_agent_rag.embeddings import create_embedding_provider
from enterprise_multi_agent_rag.generation import (
    PromptBuilder,
    RAGService,
    create_llm_service,
)
from enterprise_multi_agent_rag.retrieval import FAISSVectorStore, Retriever

router = APIRouter(tags=["query"])


class QueryRequest(BaseModel):
    """A question and the number of context chunks to retrieve."""

    question: str
    k: int = Field(default=5, gt=0)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        """Reject empty and whitespace-only questions."""
        if not value.strip():
            raise ValueError("question must not be empty or whitespace only")
        return value


class QueryResponse(BaseModel):
    """The generated answer paired with its original question."""

    question: str
    answer: str


@lru_cache
def get_rag_service() -> RAGService:
    """Build the application RAG service on first use."""
    settings = get_settings()
    retriever = Retriever(
        create_embedding_provider(settings),
        FAISSVectorStore.load("vector_store"),
    )
    return RAGService(
        retriever=retriever,
        prompt_builder=PromptBuilder(),
        llm_service=create_llm_service(settings),
    )


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> QueryResponse:
    """Answer a question using the configured RAG pipeline."""
    answer = rag_service.answer(request.question, request.k)
    return QueryResponse(question=request.question, answer=answer)
