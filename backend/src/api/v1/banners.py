from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.src.api.deps import get_db
from backend.src.core.errors.common import EntityAlreadyExistsError, EntityNotFoundError
from backend.src.core.schemas.banners import BannerCreate, BannerResponse, BannersResponse
from backend.src.repository.repo import BannerRepository
from backend.src.services.service import BannersService

router = APIRouter(prefix="/banners", tags=["banners"])


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
    except EntityAlreadyExistsError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc


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
    except EntityNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
