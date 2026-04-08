from __future__ import annotations

import httpx
from fastapi import APIRouter, Depends, HTTPException, status

from backend.src.api.deps import get_app_settings
from backend.src.core.config import Settings
from backend.src.schemas.recommendation import RecommendationRequest, RecommendationResponse

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.post("", response_model=RecommendationResponse)
async def get_recommendations(
    payload: RecommendationRequest,
    settings: Settings = Depends(get_app_settings),
) -> RecommendationResponse:
    try:
        async with httpx.AsyncClient(
            base_url=settings.inference_url,
            timeout=httpx.Timeout(30.0, connect=5.0),
        ) as client:
            response = await client.post("/recommend", json=payload.model_dump(mode="json"))
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text.strip() or exc.response.reason_phrase
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Inference service returned {exc.response.status_code}: {detail}",
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Inference service is unavailable at {settings.inference_url}: {exc}",
        ) from exc

    return RecommendationResponse.model_validate(response.json())
