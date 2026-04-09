from pydantic import BaseModel, EmailStr


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str


class User(BaseModel):
    id: int
    email: EmailStr
    full_name: str


class Users(BaseModel):
    users: list[User]