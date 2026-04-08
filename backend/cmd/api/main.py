from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.src.api.deps import get_app_settings  # noqa: E402
from backend.src.core.db import init_db  # noqa: E402


def create_app() -> FastAPI:
    settings = get_app_settings()
    app = FastAPI(title=settings.app_name)

    @app.get("/health", tags=["health"])
    def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    @app.on_event("startup")
    def on_startup() -> None:
        init_db(settings)
        
    return app


app = create_app()
