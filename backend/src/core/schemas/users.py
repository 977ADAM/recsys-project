from pydantic import BaseModel, ConfigDict, Field

from backend.src.core.schemas.schema import Gender


class UserCreate(BaseModel):
    user_id: str = Field(min_length=1, max_length=20)
    age: int = Field(ge=0, le=32767)
    gender: Gender
    city_tier: str = Field(min_length=1, max_length=10)
    device_os: str = Field(min_length=1, max_length=20)
    platform: str = Field(min_length=1, max_length=20)
    income_band: str = Field(min_length=1, max_length=10)
    activity_segment: str = Field(min_length=1, max_length=10)
    interest_1: str = Field(min_length=1, max_length=50)
    interest_2: str = Field(min_length=1, max_length=50)
    interest_3: str = Field(min_length=1, max_length=50)
    country: str = Field(min_length=2, max_length=2)
    signup_days_ago: int = Field(ge=0)
    is_premium: bool


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    user_id: str
    age: int
    gender: Gender
    city_tier: str
    device_os: str
    platform: str
    income_band: str
    activity_segment: str
    interest_1: str
    interest_2: str
    interest_3: str
    country: str
    signup_days_ago: int
    is_premium: bool


class UsersResponse(BaseModel):
    users: list[UserResponse]
