from __future__ import annotations

from collections.abc import Generator
from functools import lru_cache

from sqlalchemy.orm import Session

from backend.src.core.config import Settings, get_settings
from backend.src.core.db import get_db_session


@lru_cache(maxsize=1)
def get_app_settings() -> Settings:
    return get_settings()


def get_db() -> Generator[Session, None, None]:
    settings = get_app_settings()
    yield from get_db_session(settings)
