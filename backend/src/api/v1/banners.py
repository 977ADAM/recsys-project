from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.src.api.deps import get_db
from backend.src.core.errors.common import (
    EntityAlreadyExistsError,
    EntityNotFoundError,
    InvalidRequestError,
)
from backend.src.core.schemas.banners import BannerCreate, BannerPatch, BannerResponse, BannersResponse
from backend.src.repository.repo import BannerRepository
from backend.src.services.service import BannersService

router = APIRouter(prefix="/banners", tags=["banners"])


def _to_http_exception(exc: Exception) -> HTTPException:
    if isinstance(exc, EntityAlreadyExistsError):
        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        )
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
    response_model=BannerResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_banner(
    banner: BannerCreate,
    db: Session = Depends(get_db),
):
    service = BannersService(BannerRepository(db))

    try:
        return service.create_banner(banner)
    except (EntityAlreadyExistsError, EntityNotFoundError, InvalidRequestError) as exc:
        raise _to_http_exception(exc) from exc


@router.get(
    "",
    response_model=BannersResponse,
    status_code=status.HTTP_200_OK,
)
def get_banners(
    db: Session = Depends(get_db),
):
    service = BannersService(BannerRepository(db))
    return service.get_banners()


@router.get("/{banner_id}", response_model=BannerResponse)
def get_banner(
    banner_id: str,
    db: Session = Depends(get_db),
):
    service = BannersService(BannerRepository(db))

    try:
        return service.get_banner(banner_id)
    except (EntityAlreadyExistsError, EntityNotFoundError, InvalidRequestError) as exc:
        raise _to_http_exception(exc) from exc


@router.patch("/{banner_id}", response_model=BannerResponse, status_code=status.HTTP_200_OK)
def patch_banner(
    banner_id: str,
    banner: BannerPatch,
    db: Session = Depends(get_db),
):
    service = BannersService(BannerRepository(db))

    try:
        return service.patch_banner(banner_id, banner)
    except (EntityAlreadyExistsError, EntityNotFoundError) as exc:
        raise _to_http_exception(exc) from exc


@router.delete("/{banner_id}", response_model=BannerResponse, status_code=status.HTTP_200_OK)
def delete_banner(
    banner_id: str,
    db: Session = Depends(get_db),
):
    service = BannersService(BannerRepository(db))

    try:
        return service.delete_banner(banner_id)
    except (EntityAlreadyExistsError, EntityNotFoundError) as exc:
        raise _to_http_exception(exc) from exc
