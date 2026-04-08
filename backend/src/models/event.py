from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.src.models.model import Base


class EventRecord(Base):
    __tablename__ = "events"

    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    user_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    session_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    banner_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    page_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    event_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    stored_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    context: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

