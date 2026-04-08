from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from threading import Lock

from backend.src.core.config import Settings

try:
    import redis
except ModuleNotFoundError:
    redis = None


@dataclass
class UserBannerStats:
    served_impressions_total: float = 0.0
    served_clicks_total: float = 0.0


class InMemoryOnlineStateStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._user_seen_banners: dict[str, set[str]] = defaultdict(set)
        self._session_seen_banners: dict[str, set[str]] = defaultdict(set)
        self._user_banner_stats: dict[str, dict[str, UserBannerStats]] = defaultdict(dict)

    def record_page_view(
        self,
        user_id: str | None,
        session_id: str | None,
    ) -> None:
        # Reserved for future session-level features; page views do not mutate banner stats yet.
        _ = user_id, session_id

    def record_impression(
        self,
        user_id: str | None,
        session_id: str | None,
        banner_id: str | None,
    ) -> None:
        if banner_id is None:
            return
        with self._lock:
            if user_id:
                self._user_seen_banners[user_id].add(banner_id)
                stats = self._user_banner_stats[user_id].setdefault(banner_id, UserBannerStats())
                stats.served_impressions_total += 1.0
            if session_id:
                self._session_seen_banners[session_id].add(banner_id)

    def record_click(
        self,
        user_id: str | None,
        session_id: str | None,
        banner_id: str | None,
    ) -> None:
        if banner_id is None:
            return
        with self._lock:
            if user_id:
                self._user_seen_banners[user_id].add(banner_id)
                stats = self._user_banner_stats[user_id].setdefault(banner_id, UserBannerStats())
                stats.served_clicks_total += 1.0
            if session_id:
                self._session_seen_banners[session_id].add(banner_id)

    def get_seen_banners(
        self,
        user_id: str | None,
        session_id: str | None,
    ) -> set[str]:
        seen: set[str] = set()
        with self._lock:
            if user_id:
                seen.update(self._user_seen_banners.get(user_id, set()))
            if session_id:
                seen.update(self._session_seen_banners.get(session_id, set()))
        return seen

    def get_banner_stats(
        self,
        user_id: str | None,
        banner_ids: list[str],
    ) -> dict[str, UserBannerStats]:
        if not user_id:
            return {}
        with self._lock:
            user_stats = self._user_banner_stats.get(user_id, {})
            return {
                banner_id: UserBannerStats(
                    served_impressions_total=user_stats[banner_id].served_impressions_total,
                    served_clicks_total=user_stats[banner_id].served_clicks_total,
                )
                for banner_id in banner_ids
                if banner_id in user_stats
            }


class RedisOnlineStateStore:
    def __init__(self, redis_client: "redis.Redis") -> None:
        self.redis_client = redis_client

    def record_page_view(
        self,
        user_id: str | None,
        session_id: str | None,
    ) -> None:
        _ = user_id, session_id

    def record_impression(
        self,
        user_id: str | None,
        session_id: str | None,
        banner_id: str | None,
    ) -> None:
        if banner_id is None:
            return
        pipe = self.redis_client.pipeline()
        if user_id:
            pipe.sadd(f"user_seen:{user_id}", banner_id)
            pipe.hincrbyfloat(f"user_banner_stats:{user_id}", banner_id, 1.0)
        if session_id:
            pipe.sadd(f"session_seen:{session_id}", banner_id)
        pipe.execute()

    def record_click(
        self,
        user_id: str | None,
        session_id: str | None,
        banner_id: str | None,
    ) -> None:
        if banner_id is None:
            return
        pipe = self.redis_client.pipeline()
        if user_id:
            pipe.sadd(f"user_seen:{user_id}", banner_id)
            key = f"user_banner_clicks:{user_id}"
            pipe.hincrbyfloat(key, banner_id, 1.0)
        if session_id:
            pipe.sadd(f"session_seen:{session_id}", banner_id)
        pipe.execute()

    def get_seen_banners(
        self,
        user_id: str | None,
        session_id: str | None,
    ) -> set[str]:
        seen: set[str] = set()
        if user_id:
            seen.update(self.redis_client.smembers(f"user_seen:{user_id}"))
        if session_id:
            seen.update(self.redis_client.smembers(f"session_seen:{session_id}"))
        return {value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in seen}

    def get_banner_stats(
        self,
        user_id: str | None,
        banner_ids: list[str],
    ) -> dict[str, UserBannerStats]:
        if not user_id or not banner_ids:
            return {}
        impression_map = self.redis_client.hgetall(f"user_banner_stats:{user_id}")
        click_map = self.redis_client.hgetall(f"user_banner_clicks:{user_id}")
        stats: dict[str, UserBannerStats] = {}
        for banner_id in banner_ids:
            impression_value = impression_map.get(banner_id.encode("utf-8")) or impression_map.get(banner_id)
            click_value = click_map.get(banner_id.encode("utf-8")) or click_map.get(banner_id)
            if impression_value is None and click_value is None:
                continue
            stats[banner_id] = UserBannerStats(
                served_impressions_total=float(impression_value or 0.0),
                served_clicks_total=float(click_value or 0.0),
            )
        return stats


def build_online_state_store(settings: Settings) -> InMemoryOnlineStateStore | RedisOnlineStateStore:
    if redis is None:
        return InMemoryOnlineStateStore()
    try:
        redis_client = redis.from_url(settings.redis_url, decode_responses=False)
        redis_client.ping()
        return RedisOnlineStateStore(redis_client=redis_client)
    except Exception:
        return InMemoryOnlineStateStore()
