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
    inference_url: str
    interactions_csv: str
    users_csv: str
    banners_csv: str
    artifacts_dir: str
    retrieval_artifacts_dir: str
    event_log_path: str
    redis_url: str


def get_settings() -> Settings:
    project_root = Path(__file__).resolve().parents[3]
    return Settings(
        project_root=project_root,
        app_name=os.getenv("APP_NAME", "Recsys API"),
        api_v1_prefix=os.getenv("API_V1_PREFIX", "/api/v1"),
        database_url=os.getenv(
            "DATABASE_URL",
            "postgresql+psycopg://recsys:recsys@localhost:5433/recsys",
        ),
        inference_url=os.getenv("INFERENCE_URL", "http://localhost:8081"),
        interactions_csv=os.getenv("INTERACTIONS_CSV", "data/db/banner_interactions.csv"),
        users_csv=os.getenv("USERS_CSV", "data/db/users.csv"),
        banners_csv=os.getenv("BANNERS_CSV", "data/db/banners.csv"),
        artifacts_dir=os.getenv("ARTIFACTS_DIR", "deepfm_artifacts"),
        retrieval_artifacts_dir=os.getenv("RETRIEVAL_ARTIFACTS_DIR", "artifacts/pytorch_retrieval"),
        event_log_path=os.getenv("EVENT_LOG_PATH", "out/events/events.jsonl"),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6380/0"),
    )
