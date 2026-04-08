from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.src.core.config import Settings
from backend.src.models import event  # noqa: F401
from backend.src.models.model import Base


@lru_cache(maxsize=1)
def get_engine(database_url: str):
    return create_engine(database_url, future=True, pool_pre_ping=True)


@lru_cache(maxsize=1)
def get_session_factory(database_url: str):
    return sessionmaker(
        bind=get_engine(database_url),
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        class_=Session,
    )


def get_db_session(settings: Settings) -> Iterator[Session]:
    session_factory = get_session_factory(settings.database_url)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


def init_db(settings: Settings) -> None:
    engine = get_engine(settings.database_url)
    Base.metadata.create_all(bind=engine)
