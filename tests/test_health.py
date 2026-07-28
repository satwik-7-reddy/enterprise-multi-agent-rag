"""Tests for the health endpoint."""

from fastapi.testclient import TestClient

from enterprise_multi_agent_rag.main import app


def test_health() -> None:
    """The health endpoint reports an available service."""
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

