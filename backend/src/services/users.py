from backend.src.repository.models.users import User
from backend.src.core.schemas.users import UserCreate, Users
from backend.src.repository.repo import UserRepository
from backend.src.core.errors.common import EmailAlreadyRegisteredError
from collections.abc import Callable

PasswordHasher = Callable[[str], str]

def create_user(
    repo: UserRepository,
    user: UserCreate,
    password_hasher: PasswordHasher,
) -> User:
    existing_user = repo.get_by_email(user.email)
    if existing_user is not None:
        raise EmailAlreadyRegisteredError("Email already registered")

    return repo.create_user(
        email=user.email,
        full_name=user.full_name,
        hashed_password=password_hasher(user.password),
    )


def get_users(
    repo: UserRepository,
) -> Users:
    return repo.get_users()