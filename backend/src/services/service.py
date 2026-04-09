from backend.src.core.models.users import User
from backend.src.core.schemas.users import UserCreate
from backend.src.services.users import create_user
from backend.src.repo.users import UserRepository


class UsersService:
    def __init__(self, repo: UserRepository):
        self.repo = repo

    def Create_User(self, payload: UserCreate) -> User:
        return create_user(self.repo, payload)
