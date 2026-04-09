from src.core.security import hash_password
from backend.src.core.models.users import User
from backend.src.core.schemas.users import UserCreate

def create_user(repo, user: UserCreate) -> User:

    existing_user = repo.get_by_email(user.email)
    if existing_user:
        raise ("Email already registered")

    return repo.create(
        email=user.email,
        full_name=user.full_name,
        hashed_password=hash_password(user.password),
    )