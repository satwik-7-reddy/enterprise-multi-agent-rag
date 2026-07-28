"""FastAPI application entry point."""

from fastapi import FastAPI

from enterprise_multi_agent_rag.api.router import api_router
from enterprise_multi_agent_rag.core.config import get_settings
from enterprise_multi_agent_rag.core.logging import configure_logging


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    configure_logging(settings.log_level)

    application = FastAPI(
        title=settings.app_name,
        version="0.1.0",
    )
    application.include_router(api_router)
    return application


app = create_app()

