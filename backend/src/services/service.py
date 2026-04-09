from collections.abc import Callable

from backend.src.core.schemas.users import User, UserCreate, Users
from backend.src.repository.repo import UserRepository
from backend.src.services.users import create_user, get_users

PasswordHasher = Callable[[str], str]

class UsersService:
    def __init__(
        self,
        repo: UserRepository,
        password_hasher: PasswordHasher | None = None,
    ):
        self.repo = repo
        self.password_hasher = password_hasher

    def create_user(self, user: UserCreate) -> User:
        if self.password_hasher is None:
            raise ValueError("Password hasher is required to create users")

        return create_user(self.repo, user, self.password_hasher)

    def get_users(self) -> Users:
        return get_users(self.repo)
