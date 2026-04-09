from sqlalchemy.orm import Session

from backend.src.repository.models.users import User
from backend.src.repository.users import (
    create_user as repo_create_user,
    get_by_email as repo_get_by_email,
    get_users as repo_get_users,
)


class UserRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_email(self, email: str) -> User | None:
        return repo_get_by_email(self.db, email)

    def create_user(
        self,
        *,
        email: str,
        full_name: str,
        hashed_password: str,
    ) -> User:
        return repo_create_user(
            db=self.db,
            email=email,
            full_name=full_name,
            hashed_password=hashed_password,
        )
    
    def get_users(self) -> list[User]:
        return repo_get_users(self.db)
