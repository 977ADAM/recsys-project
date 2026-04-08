from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from uuid import uuid4

from backend.src.core.config import Settings
from backend.src.core.db import get_session_factory
from backend.src.repo.event import EventRepository
from backend.src.schemas.event import EventRequest, EventResponse
from backend.src.services.online_state import InMemoryOnlineStateStore

_event_log_lock = Lock()


class EventService:
    def __init__(self, settings: Settings, online_state: InMemoryOnlineStateStore) -> None:
        self.settings = settings
        self.online_state = online_state

    def ingest(self, payload: EventRequest) -> EventResponse:
        event_id = str(uuid4())
        stored_at_dt = datetime.now(timezone.utc)
        event_time_dt = payload.event_time or stored_at_dt
        stored_at = stored_at_dt.isoformat()
        event_time = event_time_dt.isoformat()
        event_log_path = (self.settings.project_root / self.settings.event_log_path).resolve()
        event_log_path.parent.mkdir(parents=True, exist_ok=True)

        event_record = {
            "event_id": event_id,
            "event_type": payload.event_type,
            "user_id": payload.user_id,
            "session_id": payload.session_id,
            "banner_id": payload.banner_id,
            "page_url": payload.page_url,
            "event_time": event_time,
            "stored_at": stored_at,
            "context": payload.context,
        }

        with _event_log_lock:
            with open(event_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event_record, ensure_ascii=False) + "\n")

        try:
            session_factory = get_session_factory(self.settings.database_url)
            with session_factory() as session:
                repo = EventRepository(session)
                repo.create(
                    event_id=event_id,
                    event_type=payload.event_type,
                    user_id=payload.user_id,
                    session_id=payload.session_id,
                    banner_id=payload.banner_id,
                    page_url=payload.page_url,
                    event_time=event_time_dt,
                    stored_at=stored_at_dt,
                    context=payload.context,
                )
        except Exception:
            # Keep ingestion available even if Postgres is temporarily down.
            pass

        if payload.event_type == "page_view":
            self.online_state.record_page_view(
                user_id=payload.user_id,
                session_id=payload.session_id,
            )
        elif payload.event_type == "banner_impression":
            self.online_state.record_impression(
                user_id=payload.user_id,
                session_id=payload.session_id,
                banner_id=payload.banner_id,
            )
        elif payload.event_type == "banner_click":
            self.online_state.record_click(
                user_id=payload.user_id,
                session_id=payload.session_id,
                banner_id=payload.banner_id,
            )

        return EventResponse(
            accepted=True,
            event_id=event_id,
            stored_at=stored_at,
            event_log_path=str(event_log_path),
        )
