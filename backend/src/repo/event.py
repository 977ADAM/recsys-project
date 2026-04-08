from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from backend.src.models.event import EventRecord


class EventRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(
        self,
        *,
        event_id: str,
        event_type: str,
        user_id: str | None,
        session_id: str | None,
        banner_id: str | None,
        page_url: str | None,
        event_time: datetime,
        stored_at: datetime,
        context: dict,
    ) -> EventRecord:
        record = EventRecord(
            event_id=event_id,
            event_type=event_type,
            user_id=user_id,
            session_id=session_id,
            banner_id=banner_id,
            page_url=page_url,
            event_time=event_time,
            stored_at=stored_at,
            context=context,
        )
        self.session.add(record)
        self.session.commit()
        return record
