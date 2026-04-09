from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.src.api.deps import get_db
from backend.src.core.errors.common import EmailAlreadyRegisteredError
from backend.src.core.schemas.users import UserCreate, User
from backend.src.repository.repo import UserRepository
from backend.src.services.service import UsersService

from backend.src.core.security import hash_password

router = APIRouter(prefix="/users", tags=["users"])


@router.post(
    "",
    response_model=User,
    status_code=status.HTTP_201_CREATED,
)
def create_user(
    user: UserCreate,
    db: Session = Depends(get_db),
):
    service = UsersService(
        UserRepository(db),
        password_hasher=hash_password,
    )

    try:
        return service.create_user(user)
    except EmailAlreadyRegisteredError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc