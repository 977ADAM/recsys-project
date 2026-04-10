from backend.src.core.errors.common import (
    EntityAlreadyExistsError,
    EntityNotFoundError,
    InvalidRequestError,
)
from backend.src.schemas.banners import BannerCreate, BannerPatch, BannerResponse, BannersResponse
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


def delete_banner(repo: BannerRepository, banner_id: str) -> BannerResponse:
    banner = repo.delete_banner(banner_id)
    if banner is None:
        raise EntityNotFoundError(f"Banner with banner_id={banner_id} not found")
    return BannerResponse.model_validate(banner)


def patch_banner(
    repo: BannerRepository,
    banner_id: str,
    banner: BannerPatch,
) -> BannerResponse:
    fields = banner.model_dump(exclude_unset=True)
    if "landing_page" in fields:
        fields["landing_page"] = str(fields["landing_page"])

    current_banner = repo.get_banner(banner_id)
    if current_banner is None:
        raise EntityNotFoundError(f"Banner with banner_id={banner_id} not found")

    target_age_min = fields.get("target_age_min", current_banner.target_age_min)
    target_age_max = fields.get("target_age_max", current_banner.target_age_max)
    if target_age_max < target_age_min:
        raise InvalidRequestError(
            "target_age_max must be greater than or equal to target_age_min"
        )

    updated_banner = repo.patch_banner(banner_id, **fields)
    return BannerResponse.model_validate(updated_banner)
