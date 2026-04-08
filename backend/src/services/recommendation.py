from __future__ import annotations

import httpx

from backend.src.core.config import Settings
from backend.src.schemas.recommendation import RecommendationRequest, RecommendationResponse
from backend.src.services.online_state import InMemoryOnlineStateStore, RedisOnlineStateStore


class RecommendationService:
    def __init__(
        self,
        settings: Settings,
        online_state: InMemoryOnlineStateStore | RedisOnlineStateStore,
    ) -> None:
        self.settings = settings
        self.online_state = online_state

    def recommend(self, payload: RecommendationRequest) -> RecommendationResponse:
        endpoint = f"{self.settings.inference_url.rstrip('/')}/recommend"
        online_seen_banner_ids = sorted(
            self.online_state.get_seen_banners(
                user_id=payload.user_id,
                session_id=payload.session_id,
            )
        )
        online_banner_stats = [
            {
                "banner_id": banner_id,
                "served_impressions_total": stats.served_impressions_total,
                "served_clicks_total": stats.served_clicks_total,
            }
            for banner_id, stats in self.online_state.get_banner_stats(
                user_id=payload.user_id,
                banner_ids=online_seen_banner_ids,
            ).items()
        ]
        enriched_payload = payload.model_copy(
            update={
                "online_seen_banner_ids": online_seen_banner_ids,
                "online_banner_stats": online_banner_stats,
            }
        )
        with httpx.Client(timeout=30.0) as client:
            response = client.post(endpoint, json=enriched_payload.model_dump(mode="json"))
            response.raise_for_status()
        return RecommendationResponse.model_validate(response.json())
