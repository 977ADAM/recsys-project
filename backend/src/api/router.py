from __future__ import annotations

from fastapi import APIRouter
from backend.src.api.v1 import users

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(users.router)
