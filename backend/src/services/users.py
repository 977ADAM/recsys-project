from collections.abc import Callable

from backend.src.core.errors.common import EmailAlreadyRegisteredError
from backend.src.core.schemas.users import User as UserSchema
from backend.src.core.schemas.users import UserCreate, Users
from backend.src.repository.repo import UserRepository

PasswordHasher = Callable[[str], str]


def create_user(
    repo: UserRepository,
    user: UserCreate,
    password_hasher: PasswordHasher,
) -> UserSchema:
    existing_user = repo.get_by_email(user.email)
    if existing_user is not None:
        raise EmailAlreadyRegisteredError("Email already registered")

    created_user = repo.create_user(
        email=user.email,
        full_name=user.full_name,
        hashed_password=password_hasher(user.password),
    )
    return UserSchema.model_validate(created_user)


def get_users(repo: UserRepository) -> Users:
    return Users(
        users=[UserSchema.model_validate(user) for user in repo.get_users()],
    )
