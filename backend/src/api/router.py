from __future__ import annotations

from fastapi import APIRouter

from backend.src.api.deps import get_app_settings
from backend.src.api.v1 import users

api_router = APIRouter(prefix=get_app_settings().api_v1_prefix)
api_router.include_router(users.router)