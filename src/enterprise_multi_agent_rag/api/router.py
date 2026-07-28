"""Top-level API router."""

from fastapi import APIRouter

from enterprise_multi_agent_rag.api.routes.health import router as health_router

api_router = APIRouter()
api_router.include_router(health_router)

