from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool

from backend.src.api.deps import get_app_settings, get_retrieval_service
from backend.src.core.errors.common import EntityNotFoundError, InvalidRequestError
from backend.src.schemas.recommendations import RecommendationRequest, RecommendationResponse
from backend.src.services.recommendations import recommend_banners
from backend.src.services.retrieval import RetrievalService

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


def _to_http_exception(exc: Exception) -> HTTPException:
    if isinstance(exc, EntityNotFoundError):
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    if isinstance(exc, InvalidRequestError):
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    raise exc


@router.post(
    "",
    response_model=RecommendationResponse,
    status_code=status.HTTP_200_OK,
)
async def get_recommendations(
    request: RecommendationRequest,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
):
    settings = get_app_settings()
    try:
        return await run_in_threadpool(recommend_banners, request, settings, retrieval_service)
    except (EntityNotFoundError, InvalidRequestError) as exc:
        raise _to_http_exception(exc) from exc
