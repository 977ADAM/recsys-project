from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool

from backend.src.api.deps import get_retrieval_service
from backend.src.core.errors.common import InvalidRequestError
from backend.src.schemas.retrieval import (
    RetrievalRefreshResponse,
    RetrievalReloadResponse,
    RetrievalRequest,
    RetrievalResponse,
)
from backend.src.services.retrieval import (
    RetrievalService,
    refresh_retrieval_state,
    reload_retrieval_runtime,
    retrieve_banners,
)

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


def _to_http_exception(exc: Exception) -> HTTPException:
    if isinstance(exc, InvalidRequestError):
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    raise exc


@router.post(
    "",
    response_model=RetrievalResponse,
    status_code=status.HTTP_200_OK,
)
async def get_retrieval_candidates(
    request: RetrievalRequest,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
):
    try:
        return await run_in_threadpool(retrieve_banners, request, retrieval_service)
    except InvalidRequestError as exc:
        raise _to_http_exception(exc) from exc


@router.post(
    "/refresh",
    response_model=RetrievalRefreshResponse,
    status_code=status.HTTP_200_OK,
)
async def refresh_retrieval(
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
):
    try:
        return await run_in_threadpool(refresh_retrieval_state, retrieval_service)
    except InvalidRequestError as exc:
        raise _to_http_exception(exc) from exc


@router.post(
    "/reload",
    response_model=RetrievalReloadResponse,
    status_code=status.HTTP_200_OK,
)
async def reload_retrieval(
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
):
    try:
        return await run_in_threadpool(reload_retrieval_runtime, retrieval_service)
    except InvalidRequestError as exc:
        raise _to_http_exception(exc) from exc
