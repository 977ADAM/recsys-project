from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.api.deps import get_db
from backend.src.core.schemas.users import UserCreate, UserRead
from backend.src.services.service import UsersService
from backend.src.repo.repo import UserRepository

router = APIRouter(prefix="/users", tags=["users"])

@router.post(
    "",
    response_model=UserRead,
    status_code=status.HTTP_201_CREATED,
)
def create_user(
    payload: UserCreate,
    db: Session = Depends(get_db),
):
    service = UsersService(UserRepository(db))

    try:
        return service.create_user(payload)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc















