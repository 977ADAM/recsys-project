from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class EventRequest(BaseModel):
    event_type: Literal["page_view", "banner_impression", "banner_click"]
    user_id: str | None = None
    session_id: str | None = None
    banner_id: str | None = None
    page_url: str | None = None
    event_time: datetime | None = None
    context: dict[str, Any] = Field(default_factory=dict)


class EventResponse(BaseModel):
    accepted: bool
    event_id: str
    stored_at: str
    event_log_path: str

