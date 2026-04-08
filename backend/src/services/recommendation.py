from __future__ import annotations

import httpx

from backend.src.core.config import Settings
from backend.src.schemas.recommendation import RecommendationRequest, RecommendationResponse


class RecommendationService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def recommend(self, payload: RecommendationRequest) -> RecommendationResponse:
        endpoint = f"{self.settings.inference_url.rstrip('/')}/recommend"
        with httpx.Client(timeout=30.0) as client:
            response = client.post(endpoint, json=payload.model_dump(mode="json"))
            response.raise_for_status()
        return RecommendationResponse.model_validate(response.json())
