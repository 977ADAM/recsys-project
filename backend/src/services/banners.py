from backend.src.core.errors.common import EntityAlreadyExistsError, EntityNotFoundError
from backend.src.core.schemas.banners import BannerCreate, BannerResponse, BannersResponse
from backend.src.repository.repo import BannerRepository


def create_banner(
    repo: BannerRepository,
    banner: BannerCreate,
) -> BannerResponse:
    existing_banner = repo.get_banner_by_id(banner.banner_id)
    if existing_banner is not None:
        raise EntityAlreadyExistsError(
            f"Banner with banner_id={banner.banner_id} already exists"
        )

    created_banner = repo.create_banner(
        banner_id=banner.banner_id,
        brand=banner.brand,
        category=banner.category,
        subcategory=banner.subcategory,
        banner_format=banner.banner_format,
        campaign_goal=banner.campaign_goal,
        target_gender=banner.target_gender,
        target_age_min=banner.target_age_min,
        target_age_max=banner.target_age_max,
        cpm_bid=banner.cpm_bid,
        quality_score=banner.quality_score,
        created_at=banner.created_at,
        is_active=banner.is_active,
        landing_page=str(banner.landing_page),
    )
    return BannerResponse.model_validate(created_banner)


def get_banners(repo: BannerRepository) -> BannersResponse:
    return BannersResponse(
        banners=[BannerResponse.model_validate(banner) for banner in repo.get_banners()],
    )


def get_banner(repo: BannerRepository, banner_id: str) -> BannerResponse:
    banner = repo.get_banner(banner_id)
    if banner is None:
        raise EntityNotFoundError(f"Banner with banner_id={banner_id} not found")
    return BannerResponse.model_validate(banner)
