from __future__ import annotations

from functools import lru_cache

from backend.src.core.config import Settings, get_settings
from backend.src.services.event import EventService
from backend.src.services.online_state import build_online_state_store
from backend.src.services.recommendation import RecommendationService


@lru_cache(maxsize=1)
def get_app_settings() -> Settings:
    return get_settings()


@lru_cache(maxsize=1)
def get_recommendation_service() -> RecommendationService:
    return RecommendationService(settings=get_app_settings())


@lru_cache(maxsize=1)
def get_event_service() -> EventService:
    return EventService(
        settings=get_app_settings(),
        online_state=get_online_state_store(),
    )


@lru_cache(maxsize=1)
def get_online_state_store():
    return build_online_state_store(get_app_settings())
