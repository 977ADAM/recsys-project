from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.src.api.deps import get_db
from backend.src.core.errors.common import EntityAlreadyExistsError, EntityNotFoundError
from backend.src.core.schemas.users import UserCreate, UserPatch, UserResponse, UsersResponse
from backend.src.repository.repo import UserRepository
from backend.src.services.service import UsersService

router = APIRouter(prefix="/users", tags=["users"])


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
    raise exc


@router.post(
    "",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_user(
    user: UserCreate,
    db: Session = Depends(get_db),
):
    service = UsersService(UserRepository(db))

    try:
        return service.create_user(user)
    except (EntityAlreadyExistsError, EntityNotFoundError) as exc:
        raise _to_http_exception(exc) from exc


@router.get(
    "",
    response_model=UsersResponse,
    status_code=status.HTTP_200_OK,
)
def get_users(
    db: Session = Depends(get_db),
):
    service = UsersService(UserRepository(db))
    return service.get_users()


@router.get("/{user_id}", response_model=UserResponse)
def get_user(
    user_id: str,
    db: Session = Depends(get_db),
):
    service = UsersService(UserRepository(db))
    try:
        return service.get_user(user_id)
    except (EntityAlreadyExistsError, EntityNotFoundError) as exc:
        raise _to_http_exception(exc) from exc


@router.patch("/{user_id}", response_model=UserResponse, status_code=status.HTTP_200_OK)
def patch_user(
    user_id: str,
    user: UserPatch,
    db: Session = Depends(get_db),
):
    service = UsersService(UserRepository(db))
    try:
        return service.patch_user(user_id, user)
    except (EntityAlreadyExistsError, EntityNotFoundError) as exc:
        raise _to_http_exception(exc) from exc


@router.delete("/{user_id}", response_model=UserResponse, status_code=status.HTTP_200_OK)
def delete_user(user_id: str, db: Session = Depends(get_db)):
    service = UsersService(UserRepository(db))
    try:
        return service.delete_user(user_id)
    except (EntityAlreadyExistsError, EntityNotFoundError) as exc:
        raise _to_http_exception(exc) from exc
