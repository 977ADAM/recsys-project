from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


RetrievalSource = Literal["two_tower", "popular_fallback"]


class RetrievalRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=20)
    top_k: int = Field(default=100, ge=1, le=1000)
    exclude_seen: bool = False
    only_active: bool = False
    interactions_csv: str | None = None
    banners_csv: str | None = None
    artifacts_dir: str | None = None


class RetrievalItem(BaseModel):
    banner_id: str
    retrieval_rank: int
    retrieval_score: float


class RetrievalResponse(BaseModel):
    user_id: str
    source: RetrievalSource
    model_version: str
    items: list[RetrievalItem]


class RetrievalRefreshResponse(BaseModel):
    model_version: str
    active_banner_count: int
    popular_banner_count: int
    seen_user_count: int


class RetrievalReloadResponse(BaseModel):
    model_version: str
    embedding_dim: int
    num_users: int
    num_items: int
    active_banner_count: int
    popular_banner_count: int
    seen_user_count: int
