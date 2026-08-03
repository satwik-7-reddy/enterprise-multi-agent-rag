"""Tests for the question-answering endpoint."""

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from enterprise_multi_agent_rag.api.routes.query import get_rag_service
from enterprise_multi_agent_rag.main import app


@pytest.fixture
def rag_service() -> Mock:
    """Provide a RAG service mock without initializing external services."""
    service = Mock()
    service.answer.return_value = "Employees receive 20 vacation days."
    return service


@pytest.fixture
def client(rag_service: Mock) -> TestClient:
    """Create a client whose query dependency uses the mock service."""
    app.dependency_overrides[get_rag_service] = lambda: rag_service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_successful_request(client: TestClient) -> None:
    response = client.post("/query", json={"question": "How many vacation days?", "k": 5})

    assert response.status_code == 200
    assert response.json() == {
        "question": "How many vacation days?",
        "answer": "Employees receive 20 vacation days.",
    }


def test_default_k(client: TestClient, rag_service: Mock) -> None:
    client.post("/query", json={"question": "How many vacation days?"})

    rag_service.answer.assert_called_once_with("How many vacation days?", 5)


def test_custom_k(client: TestClient, rag_service: Mock) -> None:
    client.post("/query", json={"question": "How many vacation days?", "k": 3})

    rag_service.answer.assert_called_once_with("How many vacation days?", 3)


@pytest.mark.parametrize("question", ["", "   "])
def test_invalid_question(client: TestClient, rag_service: Mock, question: str) -> None:
    response = client.post("/query", json={"question": question})

    assert response.status_code == 422
    rag_service.answer.assert_not_called()


@pytest.mark.parametrize("k", [0, -1])
def test_invalid_k(client: TestClient, rag_service: Mock, k: int) -> None:
    response = client.post("/query", json={"question": "A valid question?", "k": k})

    assert response.status_code == 422
    rag_service.answer.assert_not_called()


def test_health_still_works(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
