from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.src.api.deps import get_recommendation_service
from backend.src.schemas.recommendation import RecommendationRequest, RecommendationResponse
from backend.src.services.recommendation import RecommendationService

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.post("", response_model=RecommendationResponse)
def recommend(
    payload: RecommendationRequest,
    service: RecommendationService = Depends(get_recommendation_service),
) -> RecommendationResponse:
    return service.recommend(payload)

