from backend.src.core.models.users import User
from backend.src.core.schemas.users import UserCreate
from backend.src.services.users import create_user
from backend.src.repository.repo import UserRepository


class UsersService:
    def __init__(self, repo: UserRepository):
        self.repo = repo

    def create_user(self, user: UserCreate) -> User:
        return create_user(self.repo, user)