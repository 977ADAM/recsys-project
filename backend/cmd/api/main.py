from __future__ import annotations

from contextlib import asynccontextmanager
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.src.api.deps import get_app_settings  # noqa: E402
from backend.src.api.router import api_router  # noqa: E402
from backend.src.core.db import init_db  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_app_settings()
    init_db(settings)
    yield


def create_app() -> FastAPI:
    settings = get_app_settings()
    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    frontend_dir = PROJECT_ROOT / "frontend"

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)

    @app.get("/health", tags=["health"])
    async def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    if frontend_dir.exists():
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
    return app


app = create_app()
