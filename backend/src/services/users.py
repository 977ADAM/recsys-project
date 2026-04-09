from backend.src.core.errors.common import EntityAlreadyExistsError, EntityNotFoundError
from backend.src.core.schemas.users import UserCreate, UserPatch, UserResponse, UsersResponse
from backend.src.repository.repo import UserRepository


def create_user(
    repo: UserRepository,
    user: UserCreate,
) -> UserResponse:
    existing_user = repo.get_by_user_id(user.user_id)
    if existing_user is not None:
        raise EntityAlreadyExistsError(f"User with user_id={user.user_id} already exists")

    created_user = repo.create_user(
        user_id=user.user_id,
        age=user.age,
        gender=user.gender,
        city_tier=user.city_tier,
        device_os=user.device_os,
        platform=user.platform,
        income_band=user.income_band,
        activity_segment=user.activity_segment,
        interest_1=user.interest_1,
        interest_2=user.interest_2,
        interest_3=user.interest_3,
        country=user.country,
        signup_days_ago=user.signup_days_ago,
        is_premium=user.is_premium,
    )
    return UserResponse.model_validate(created_user)


def get_users(repo: UserRepository) -> UsersResponse:
    users = [UserResponse.model_validate(user) for user in repo.get_users()]
    return UsersResponse(users=users)

def get_user(repo: UserRepository, user_id: str) -> UserResponse:
    user = repo.get_user(user_id)
    if user is None:
        raise EntityNotFoundError(f"User with user_id={user_id} not found")
    return UserResponse.model_validate(user)

def delete_user(repo: UserRepository, user_id: str) -> UserResponse:
    user = repo.delete_user(user_id)
    if user is None:
        raise EntityNotFoundError(f"User with user_id={user_id} not found")
    return UserResponse.model_validate(user)


def patch_user(
    repo: UserRepository,
    user_id: str,
    user: UserPatch,
) -> UserResponse:
    updated_user = repo.patch_user(user_id, **user.model_dump(exclude_unset=True))
    if updated_user is None:
        raise EntityNotFoundError(f"User with user_id={user_id} not found")
    return UserResponse.model_validate(updated_user)
