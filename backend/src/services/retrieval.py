from __future__ import annotations

import json
import logging
from pathlib import Path
from time import perf_counter

from redis import Redis
from redis.exceptions import RedisError

from backend.src.core.config import Settings
from backend.src.core.errors.common import InvalidRequestError
from backend.src.schemas.retrieval import (
    RetrievalItem,
    RetrievalRefreshResponse,
    RetrievalReloadResponse,
    RetrievalRequest,
    RetrievalResponse,
)
from src.retrieval.monitoring.alerts import ensure_candidates_available
from src.retrieval.monitoring.metrics import (
    clear_state_metric_caches,
    load_active_banner_ids,
    load_popular_banner_scores,
    load_seen_banner_history,
    merge_seen_histories,
)
from src.retrieval.serving.api import build_fallback_candidates, search_top_k
from src.retrieval.serving.loader import clear_runtime_caches, load_runtime
from src.retrieval.serving.schemas import (
    RetrievalResult,
    RetrievalRuntime,
    RetrievalState,
)
from src.retrieval.utils.common import resolve_project_path

DEFAULT_INTERACTIONS = "data/db/banner_interactions.csv"
DEFAULT_BANNERS = "data/db/banners.csv"
DEFAULT_RETRIEVAL_ARTIFACTS = "artifacts/pytorch_retrieval"
REDIS_ACTIVE_BANNERS_KEY = "retrieval:active_banners"
REDIS_POPULAR_BANNERS_KEY = "retrieval:popular_banners"
REDIS_SEEN_HASH_KEY = "retrieval:seen_banners"
REDIS_LIVE_SEEN_HASH_KEY = "retrieval:live_seen_banners"

logger = logging.getLogger(__name__)


def _load_metric(loader, *args):
    try:
        return loader(*args)
    except ValueError as exc:
        raise InvalidRequestError(str(exc)) from exc


class RetrievalService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.default_artifact_dir = resolve_project_path(
            settings.project_root,
            None,
            DEFAULT_RETRIEVAL_ARTIFACTS,
        )
        self.default_banners_csv = resolve_project_path(
            settings.project_root,
            None,
            DEFAULT_BANNERS,
        )
        self.default_interactions_csv = resolve_project_path(
            settings.project_root,
            None,
            DEFAULT_INTERACTIONS,
        )
        self.state: RetrievalState | None = None
        self.live_seen_banner_history: dict[str, set[str]] = {}
        self.redis_client = self._build_redis_client()

    def _build_redis_client(self) -> Redis | None:
        try:
            client = Redis.from_url(self.settings.redis_url, decode_responses=True)
            client.ping()
            logger.info("retrieval redis connected url=%s", self.settings.redis_url)
            return client
        except RedisError as exc:
            logger.warning(
                "retrieval redis unavailable url=%s fallback=in_memory error=%s",
                self.settings.redis_url,
                exc,
            )
            return None

    def _redis_available(self) -> bool:
        return self.redis_client is not None

    def _write_state_to_redis(self, state: RetrievalState) -> None:
        if not self._redis_available():
            return

        assert self.redis_client is not None
        try:
            pipeline = self.redis_client.pipeline()
            pipeline.delete(REDIS_ACTIVE_BANNERS_KEY)
            if state.active_banner_ids:
                pipeline.sadd(REDIS_ACTIVE_BANNERS_KEY, *sorted(state.active_banner_ids))

            pipeline.delete(REDIS_POPULAR_BANNERS_KEY)
            if state.popular_banner_scores:
                mapping = {
                    banner_id: score
                    for banner_id, score in state.popular_banner_scores
                }
                pipeline.zadd(REDIS_POPULAR_BANNERS_KEY, mapping)

            pipeline.delete(REDIS_SEEN_HASH_KEY)
            if state.seen_banner_history:
                pipeline.hset(
                    REDIS_SEEN_HASH_KEY,
                    mapping={
                        user_id: json.dumps(sorted(banner_ids))
                        for user_id, banner_ids in state.seen_banner_history.items()
                    },
                )

            pipeline.delete(REDIS_LIVE_SEEN_HASH_KEY)
            if self.live_seen_banner_history:
                pipeline.hset(
                    REDIS_LIVE_SEEN_HASH_KEY,
                    mapping={
                        user_id: json.dumps(sorted(banner_ids))
                        for user_id, banner_ids in self.live_seen_banner_history.items()
                    },
                )
            pipeline.execute()
        except RedisError as exc:
            logger.warning("retrieval redis state write failed error=%s", exc)

    def _read_active_banner_ids_from_redis(self) -> set[str] | None:
        if not self._redis_available():
            return None
        assert self.redis_client is not None
        try:
            banner_ids = self.redis_client.smembers(REDIS_ACTIVE_BANNERS_KEY)
            return set(banner_ids)
        except RedisError as exc:
            logger.warning("retrieval redis active read failed error=%s", exc)
            return None

    def _read_popular_banner_scores_from_redis(self) -> list[tuple[str, float]] | None:
        if not self._redis_available():
            return None
        assert self.redis_client is not None
        try:
            entries = self.redis_client.zrevrange(
                REDIS_POPULAR_BANNERS_KEY,
                0,
                -1,
                withscores=True,
            )
            return [(str(banner_id), float(score)) for banner_id, score in entries]
        except RedisError as exc:
            logger.warning("retrieval redis popular read failed error=%s", exc)
            return None

    def _read_seen_banner_ids_from_redis(self, user_id: str) -> set[str] | None:
        if not self._redis_available():
            return None
        assert self.redis_client is not None
        try:
            raw_value = self.redis_client.hget(REDIS_SEEN_HASH_KEY, user_id)
            if raw_value is None:
                return set()
            banner_ids = json.loads(raw_value)
            return {str(banner_id) for banner_id in banner_ids}
        except (RedisError, json.JSONDecodeError) as exc:
            logger.warning("retrieval redis seen read failed user_id=%s error=%s", user_id, exc)
            return None

    def _read_seen_banner_history_hash_from_redis(
        self,
        redis_key: str,
        log_label: str,
    ) -> dict[str, set[str]] | None:
        if not self._redis_available():
            return None
        assert self.redis_client is not None
        try:
            raw_mapping = self.redis_client.hgetall(redis_key)
            return {
                str(user_id): {str(banner_id) for banner_id in json.loads(raw_banner_ids)}
                for user_id, raw_banner_ids in raw_mapping.items()
            }
        except (RedisError, json.JSONDecodeError) as exc:
            logger.warning("retrieval redis %s read failed error=%s", log_label, exc)
            return None

    def _read_seen_banner_history_from_redis(self) -> dict[str, set[str]] | None:
        return self._read_seen_banner_history_hash_from_redis(
            REDIS_SEEN_HASH_KEY,
            "seen history",
        )

    def _read_live_seen_banner_history_from_redis(self) -> dict[str, set[str]] | None:
        return self._read_seen_banner_history_hash_from_redis(
            REDIS_LIVE_SEEN_HASH_KEY,
            "live seen history",
        )

    def _write_seen_banner_ids_to_redis(self, user_id: str, banner_ids: set[str]) -> None:
        if not self._redis_available():
            return
        assert self.redis_client is not None
        try:
            self.redis_client.hset(
                REDIS_SEEN_HASH_KEY,
                user_id,
                json.dumps(sorted(banner_ids)),
            )
        except RedisError as exc:
            logger.warning("retrieval redis seen write failed user_id=%s error=%s", user_id, exc)

    def _write_live_seen_banner_ids_to_redis(self, user_id: str, banner_ids: set[str]) -> None:
        if not self._redis_available():
            return
        assert self.redis_client is not None
        try:
            self.redis_client.hset(
                REDIS_LIVE_SEEN_HASH_KEY,
                user_id,
                json.dumps(sorted(banner_ids)),
            )
        except RedisError as exc:
            logger.warning("retrieval redis live seen write failed user_id=%s error=%s", user_id, exc)

    def load(self) -> None:
        started_at = perf_counter()
        runtime = self._get_runtime(None)
        self.refresh_state(force=True)
        logger.info(
            "retrieval runtime loaded model_version=%s artifact_dir=%s num_users=%s num_items=%s load_ms=%.2f",
            runtime.model_version,
            runtime.artifact_dir,
            len(runtime.user2idx),
            len(runtime.item2idx),
            (perf_counter() - started_at) * 1000,
        )

    def refresh_state(self, force: bool = False) -> RetrievalState:
        if force:
            clear_state_metric_caches()

        active_banner_ids: set[str] = set()
        popular_banner_scores: list[tuple[str, float]] = []
        seen_banner_history: dict[str, set[str]] = {}

        if self.default_banners_csv.exists():
            active_banner_ids = _load_metric(load_active_banner_ids, str(self.default_banners_csv))
        if self.default_interactions_csv.exists():
            popular_banner_scores = _load_metric(
                load_popular_banner_scores,
                str(self.default_interactions_csv),
            )
            seen_banner_history = _load_metric(
                load_seen_banner_history,
                str(self.default_interactions_csv),
            )

        redis_live_seen_banner_history = self._read_live_seen_banner_history_from_redis()
        if redis_live_seen_banner_history is not None:
            self.live_seen_banner_history = merge_seen_histories(
                redis_live_seen_banner_history,
                self.live_seen_banner_history,
            )

        seen_banner_history = merge_seen_histories(
            seen_banner_history,
            self.live_seen_banner_history,
        )

        self.state = RetrievalState(
            active_banner_ids=active_banner_ids,
            popular_banner_scores=popular_banner_scores,
            seen_banner_history=seen_banner_history,
        )
        self._write_state_to_redis(self.state)
        logger.info(
            "retrieval state refreshed active_banner_count=%s popular_banner_count=%s seen_user_count=%s redis_enabled=%s",
            len(active_banner_ids),
            len(popular_banner_scores),
            len(seen_banner_history),
            self._redis_available(),
        )
        return self.state

    def _get_state(self) -> RetrievalState:
        if self.state is None:
            self.refresh_state()
        return self.state

    def record_served_banners(self, user_id: str, banner_ids: list[str]) -> None:
        if not banner_ids:
            return

        state = self._get_state()
        current_seen = set(state.seen_banner_history.get(user_id, set()))
        current_seen.update(str(banner_id) for banner_id in banner_ids)
        state.seen_banner_history[user_id] = current_seen
        current_live_seen = set(self.live_seen_banner_history.get(user_id, set()))
        current_live_seen.update(str(banner_id) for banner_id in banner_ids)
        self.live_seen_banner_history[user_id] = current_live_seen
        self._write_seen_banner_ids_to_redis(user_id, current_seen)
        self._write_live_seen_banner_ids_to_redis(user_id, current_live_seen)
        logger.info(
            "retrieval seen updated user_id=%s added_count=%s total_seen_count=%s redis_enabled=%s",
            user_id,
            len(banner_ids),
            len(current_seen),
            self._redis_available(),
        )

    def _get_runtime(self, artifacts_dir: str | None) -> RetrievalRuntime:
        artifact_path = resolve_project_path(
            self.settings.project_root,
            artifacts_dir,
            DEFAULT_RETRIEVAL_ARTIFACTS,
        )
        return load_runtime(str(artifact_path))

    def _get_seen_items(
        self,
        request: RetrievalRequest,
        runtime: RetrievalRuntime,
    ) -> set[int]:
        if not request.exclude_seen:
            return set()

        interactions_csv = resolve_project_path(
            self.settings.project_root,
            request.interactions_csv,
            DEFAULT_INTERACTIONS,
        )
        if interactions_csv == self.default_interactions_csv:
            seen_banner_ids = self._read_seen_banner_ids_from_redis(request.user_id)
            if seen_banner_ids is None:
                seen_banner_ids = self._get_state().seen_banner_history.get(request.user_id, set())
        else:
            seen_history = _load_metric(load_seen_banner_history, str(interactions_csv))
            seen_banner_ids = seen_history.get(request.user_id, set())
        return {
            runtime.item2idx[banner_id]
            for banner_id in seen_banner_ids
            if banner_id in runtime.item2idx
        }

    def _get_active_banner_ids(self, request: RetrievalRequest) -> set[str] | None:
        if not request.only_active:
            return None

        banners_csv = resolve_project_path(
            self.settings.project_root,
            request.banners_csv,
            DEFAULT_BANNERS,
        )
        if banners_csv == self.default_banners_csv:
            active_banner_ids = self._read_active_banner_ids_from_redis()
            if active_banner_ids is not None and active_banner_ids:
                return active_banner_ids
            return self._get_state().active_banner_ids
        return _load_metric(load_active_banner_ids, str(banners_csv))

    def _fallback_candidates(
        self,
        request: RetrievalRequest,
        active_banner_ids: set[str] | None,
    ):
        interactions_csv = resolve_project_path(
            self.settings.project_root,
            request.interactions_csv,
            DEFAULT_INTERACTIONS,
        )
        if interactions_csv == self.default_interactions_csv:
            popular = self._read_popular_banner_scores_from_redis()
            if popular is None or not popular:
                popular = self._get_state().popular_banner_scores
        else:
            popular = _load_metric(load_popular_banner_scores, str(interactions_csv))
        return build_fallback_candidates(popular, request.top_k, active_banner_ids)

    def get_candidates(self, request: RetrievalRequest) -> RetrievalResult:
        started_at = perf_counter()
        runtime = self._get_runtime(request.artifacts_dir)
        active_banner_ids = self._get_active_banner_ids(request)

        if request.user_id in runtime.user2idx:
            seen_items = self._get_seen_items(request, runtime)
            items = search_top_k(
                runtime,
                request.user_id,
                request.top_k,
                seen_items,
                active_banner_ids,
            )
            source = "two_tower"
        else:
            items = self._fallback_candidates(request, active_banner_ids)
            source = "popular_fallback"

        try:
            ensure_candidates_available(len(items))
        except ValueError as exc:
            raise InvalidRequestError(str(exc)) from exc

        result = RetrievalResult(
            user_id=request.user_id,
            source=source,
            model_version=runtime.model_version,
            items=items,
        )
        logger.info(
            "retrieval request user_id=%s source=%s top_k=%s exclude_seen=%s only_active=%s candidate_count=%s model_version=%s latency_ms=%.2f",
            request.user_id,
            source,
            request.top_k,
            request.exclude_seen,
            request.only_active,
            len(items),
            runtime.model_version,
            (perf_counter() - started_at) * 1000,
        )
        return result

    def refresh(self) -> RetrievalRefreshResponse:
        started_at = perf_counter()
        runtime = self._get_runtime(None)
        state = self.refresh_state(force=True)
        logger.info(
            "retrieval refresh completed model_version=%s active_banner_count=%s popular_banner_count=%s seen_user_count=%s latency_ms=%.2f",
            runtime.model_version,
            len(state.active_banner_ids),
            len(state.popular_banner_scores),
            len(state.seen_banner_history),
            (perf_counter() - started_at) * 1000,
        )
        return RetrievalRefreshResponse(
            model_version=runtime.model_version,
            active_banner_count=len(state.active_banner_ids),
            popular_banner_count=len(state.popular_banner_scores),
            seen_user_count=len(state.seen_banner_history),
        )

    def reload(self) -> RetrievalReloadResponse:
        started_at = perf_counter()
        clear_runtime_caches()
        clear_state_metric_caches()
        runtime = self._get_runtime(None)
        state = self.refresh_state(force=False)
        logger.info(
            "retrieval reload completed model_version=%s num_users=%s num_items=%s embedding_dim=%s active_banner_count=%s popular_banner_count=%s seen_user_count=%s latency_ms=%.2f",
            runtime.model_version,
            len(runtime.user2idx),
            len(runtime.item2idx),
            runtime.embedding_dim,
            len(state.active_banner_ids),
            len(state.popular_banner_scores),
            len(state.seen_banner_history),
            (perf_counter() - started_at) * 1000,
        )
        return RetrievalReloadResponse(
            model_version=runtime.model_version,
            embedding_dim=runtime.embedding_dim,
            num_users=len(runtime.user2idx),
            num_items=len(runtime.item2idx),
            active_banner_count=len(state.active_banner_ids),
            popular_banner_count=len(state.popular_banner_scores),
            seen_user_count=len(state.seen_banner_history),
        )


def retrieve_banners(
    request: RetrievalRequest,
    service: RetrievalService,
) -> RetrievalResponse:
    result = service.get_candidates(request)
    return RetrievalResponse(
        user_id=result.user_id,
        source=result.source,
        model_version=result.model_version,
        items=[
            RetrievalItem(
                banner_id=item.banner_id,
                retrieval_rank=item.retrieval_rank,
                retrieval_score=item.retrieval_score,
            )
            for item in result.items
        ],
    )


def refresh_retrieval_state(service: RetrievalService) -> RetrievalRefreshResponse:
    return service.refresh()


def reload_retrieval_runtime(service: RetrievalService) -> RetrievalReloadResponse:
    return service.reload()
