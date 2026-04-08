from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.src.api.deps import get_app_settings  # noqa: E402
from backend.src.api.router import api_router  # noqa: E402


def create_app() -> FastAPI:
    settings = get_app_settings()
    app = FastAPI(title=settings.app_name)

    @app.get("/health", tags=["health"])
    def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(api_router, prefix=settings.api_v1_prefix)
    return app


app = create_app()
