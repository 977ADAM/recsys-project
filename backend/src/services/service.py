from backend.src.core.schemas.banners import BannerCreate, BannerResponse, BannersResponse
from backend.src.core.schemas.users import UserResponse, UserCreate, UsersResponse
from backend.src.repository.repo import BannerRepository, UserRepository
from backend.src.services.banners import create_banner, get_banner, get_banners
from backend.src.services.users import create_user, get_user, get_users


class UsersService:
    def __init__(self, repo: UserRepository):
        self.repo = repo

    def create_user(self, user: UserCreate) -> UserResponse:
        return create_user(self.repo, user)

    def get_users(self) -> UsersResponse:
        return get_users(self.repo)

    def get_user(self, user_id: str) -> UserResponse:
        return get_user(self.repo, user_id)


class BannersService:
    def __init__(self, repo: BannerRepository):
        self.repo = repo

    def create_banner(self, banner: BannerCreate) -> BannerResponse:
        return create_banner(self.repo, banner)

    def get_banners(self) -> BannersResponse:
        return get_banners(self.repo)

    def get_banner(self, banner_id: str) -> BannerResponse:
        return get_banner(self.repo, banner_id)
