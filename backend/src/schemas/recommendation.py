from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    user_id: str = Field(..., description="Identifier of the user to recommend banners for.")
    top_k: int = Field(default=10, ge=1, le=100)
    score_mode: Literal["ctr", "value"] = "ctr"
    only_active: bool = True
    exclude_seen: bool = True
    as_of_date: str | None = Field(default=None, description="Optional YYYY-MM-DD serve date.")
    candidate_mode: Literal["all banners", "retrieval + ranking"] = "retrieval + ranking"
    retrieval_top_n: int = Field(default=100, ge=1, le=1000)


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
    top_k: int
    items: list[RecommendationItem]

