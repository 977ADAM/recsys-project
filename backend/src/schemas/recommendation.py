from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class OnlineBannerStats(BaseModel):
    banner_id: str
    served_impressions_total: float = 0.0
    served_clicks_total: float = 0.0


class RecommendationRequest(BaseModel):
    user_id: str = Field(..., examples=["u_00007"])
    top_k: int = Field(default=10, ge=1, le=50)
    as_of_date: str | None = None
    score_mode: Literal["ctr", "value"] = "value"
    only_active: bool = True
    exclude_seen: bool = True
    candidate_mode: Literal["all banners", "retrieval + ranking"] = "retrieval + ranking"
    retrieval_top_n: int = Field(default=100, ge=1, le=500)
    online_seen_banner_ids: list[str] = Field(default_factory=list)
    online_banner_stats: list[OnlineBannerStats] = Field(default_factory=list)


class RecommendationItem(BaseModel):
    banner_id: str
    brand: str
    category: str
    subcategory: str
    banner_format: str
    campaign_goal: str
    pred_ctr: float
    final_score: float
    cpm_bid: float
    quality_score: float
    age_match: int
    gender_match: int
    interest_match_any: int
    interest_match_count: int
    banner_ctr_prior: float
    user_ctr_prior: float
    user_subcategory_ctr_prior: float
    user_banner_ctr_prior: float
    served_impressions_total: float
    served_clicks_total: float
    is_active: int


class RecommendationResponse(BaseModel):
    model_type: str
    artifacts_dir: str
    retrieval_used: bool
    online_state_applied: bool
    top_k: int
    items: list[RecommendationItem]
