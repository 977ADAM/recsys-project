from sqlalchemy import select
from sqlalchemy.orm import Session

from src.core.models.users import User




def get_by_email(repo, email: str) -> User | None:
    return self.db.execute(
        select(User).where(User.email == email)
    ).scalar_one_or_none()

def create(
    self,
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
    self.db.add(user)
    self.db.commit()
    self.db.refresh(user)
    return user