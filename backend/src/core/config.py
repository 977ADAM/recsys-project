from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    project_root: Path
    app_name: str
    api_v1_prefix: str
    database_url: str


def get_settings() -> Settings:
    project_root = Path(__file__).resolve().parents[3]
    return Settings(
        project_root=project_root,
        app_name=os.getenv("APP_NAME", "Recsys API"),
        api_v1_prefix=os.getenv("API_V1_PREFIX", "/api/v1"),
        database_url=os.getenv(
            "DATABASE_URL",
            "postgresql+psycopg://recsys:recsys@localhost:5432/recsys",
        ),
    )
