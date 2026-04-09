from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.src.repository.models.users import User

def get_by_email(db: Session, email: str) -> User | None:
    return db.execute(
        select(User).where(User.email == email)
    ).scalar_one_or_none()

def create_user(
    db: Session,
    *,
    email: str,
    full_name: str,
    hashed_password: str,
) -> User:
    user = User(
        email=email,
        full_name=full_name,
        hashed_password=hashed_password,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

