from collections.abc import Callable

from backend.src.core.models.users import User
from backend.src.core.schemas.users import UserCreate
from backend.src.services.users import create_user
from backend.src.repository.repo import UserRepository

PasswordHasher = Callable[[str], str]

class UsersService:
    def __init__(self, repo: UserRepository, password_hasher: PasswordHasher):
        self.repo = repo
        self.password_hasher = password_hasher

    def create_user(self, user: UserCreate) -> User:
        return create_user(self.repo, user, self.password_hasher)