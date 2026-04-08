from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.src.api.deps import get_event_service
from backend.src.schemas.event import EventRequest, EventResponse
from backend.src.services.event import EventService

router = APIRouter(prefix="/events", tags=["events"])


@router.post("", response_model=EventResponse)
def ingest_event(
    payload: EventRequest,
    service: EventService = Depends(get_event_service),
) -> EventResponse:
    return service.ingest(payload)

