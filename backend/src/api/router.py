from __future__ import annotations

from fastapi import APIRouter

from backend.src.api.v1.recommendations import router as recommendations_router

api_router = APIRouter()
api_router.include_router(recommendations_router)

