from backend.src.core.schemas.banners import (
    BannerCreate,
    BannerPatch,
    BannerResponse,
    BannersResponse,
)
from backend.src.core.schemas.users import UserCreate, UserPatch, UserResponse, UsersResponse
from backend.src.repository.repo import BannerRepository, UserRepository
from backend.src.services import banners as banners_service
from backend.src.services import users as users_service


class UsersService:
    def __init__(self, repo: UserRepository):
        self.repo = repo

    def create_user(self, user: UserCreate) -> UserResponse:
        return users_service.create_user(self.repo, user)

    def get_users(self) -> UsersResponse:
        return users_service.get_users(self.repo)

    def get_user(self, user_id: str) -> UserResponse:
        return users_service.get_user(self.repo, user_id)

    def delete_user(self, user_id: str) -> UserResponse:
        return users_service.delete_user(self.repo, user_id)

    def patch_user(self, user_id: str, user: UserPatch) -> UserResponse:
        return users_service.patch_user(self.repo, user_id, user)


class BannersService:
    def __init__(self, repo: BannerRepository):
        self.repo = repo

    def create_banner(self, banner: BannerCreate) -> BannerResponse:
        return banners_service.create_banner(self.repo, banner)

    def get_banners(self) -> BannersResponse:
        return banners_service.get_banners(self.repo)

    def get_banner(self, banner_id: str) -> BannerResponse:
        return banners_service.get_banner(self.repo, banner_id)

    def delete_banner(self, banner_id: str) -> BannerResponse:
        return banners_service.delete_banner(self.repo, banner_id)

    def patch_banner(self, banner_id: str, banner: BannerPatch) -> BannerResponse:
        return banners_service.patch_banner(self.repo, banner_id, banner)
