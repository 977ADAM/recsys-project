from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from redis import Redis
from redis.exceptions import RedisError
import torch

from backend.src.core.config import Settings
from backend.src.core.errors.common import InvalidRequestError
from backend.src.schemas.retrieval import (
    RetrievalItem,
    RetrievalRefreshResponse,
    RetrievalReloadResponse,
    RetrievalRequest,
    RetrievalResponse,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = PROJECT_ROOT / "src" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from src.scripts.pytorch_recsys.inference import (  # noqa: E402
    load_item_embeddings,
    load_retrieval_model,
)

DEFAULT_INTERACTIONS = "data/db/banner_interactions.csv"
DEFAULT_BANNERS = "data/db/banners.csv"
DEFAULT_RETRIEVAL_ARTIFACTS = "artifacts/pytorch_retrieval"
REDIS_ACTIVE_BANNERS_KEY = "retrieval:active_banners"
REDIS_POPULAR_BANNERS_KEY = "retrieval:popular_banners"
REDIS_SEEN_HASH_KEY = "retrieval:seen_banners"

logger = logging.getLogger(__name__)


def _resolve_path(project_root: Path, raw_path: str | None, fallback: str) -> Path:
    path = Path(raw_path or fallback)
    if not path.is_absolute():
        path = project_root / path
    return path


@dataclass(frozen=True)
class RetrievalRuntime:
    artifact_dir: Path
    model: object
    user2idx: dict[str, int]
    item2idx: dict[str, int]
    idx2item: dict[int, str]
    embedding_dim: int
    device: torch.device
    item_embeddings: torch.Tensor
    metadata: dict
    model_version: str


@dataclass(frozen=True)
class RetrievalCandidate:
    banner_id: str
    retrieval_rank: int
    retrieval_score: float


@dataclass(frozen=True)
class RetrievalResult:
    user_id: str
    source: str
    model_version: str
    items: list[RetrievalCandidate]


@dataclass
class RetrievalState:
    active_banner_ids: set[str]
    popular_banner_scores: list[tuple[str, float]]
    seen_banner_history: dict[str, set[str]]


@lru_cache(maxsize=8)
def _load_runtime(artifact_dir: str) -> RetrievalRuntime:
    artifact_path = Path(artifact_dir)
    metadata_path = artifact_path / "metadata.json"
    if not metadata_path.exists():
        raise InvalidRequestError(
            f"Retrieval artifacts are incomplete: {metadata_path} was not found."
        )

    with metadata_path.open("r", encoding="utf-8") as file_obj:
        metadata = json.load(file_obj)

    model, user2idx, item2idx, idx2item, embedding_dim, device = load_retrieval_model(
        artifact_dir=str(artifact_path),
        device=torch.device("cpu"),
    )
    item_embeddings = load_item_embeddings(
        artifact_dir=str(artifact_path),
        model=model,
        num_items=len(item2idx),
        device=device,
    )

    model_version = str(metadata.get("model_version") or artifact_path.name)
    return RetrievalRuntime(
        artifact_dir=artifact_path,
        model=model,
        user2idx=user2idx,
        item2idx=item2idx,
        idx2item=idx2item,
        embedding_dim=embedding_dim,
        device=device,
        item_embeddings=item_embeddings,
        metadata=metadata,
        model_version=model_version,
    )


@lru_cache(maxsize=8)
def _load_active_banner_ids(banners_csv: str) -> set[str]:
    banners = pd.read_csv(banners_csv)
    if "banner_id" not in banners.columns or "is_active" not in banners.columns:
        raise InvalidRequestError(
            "banners_csv must contain banner_id and is_active columns."
        )
    active_ids = banners.loc[banners["is_active"] == 1, "banner_id"].astype(str)
    return set(active_ids.tolist())


@lru_cache(maxsize=8)
def _load_popular_banner_scores(interactions_csv: str) -> list[tuple[str, float]]:
    interactions = pd.read_csv(interactions_csv)
    if "banner_id" not in interactions.columns or "clicks" not in interactions.columns:
        raise InvalidRequestError(
            "interactions_csv must contain banner_id and clicks columns."
        )

    popularity = (
        interactions.groupby("banner_id", as_index=False)
        .agg(
            clicks=("clicks", "sum"),
            impressions=("impressions", "sum"),
        )
        .sort_values(["clicks", "impressions", "banner_id"], ascending=[False, False, True])
    )
    return [
        (str(row.banner_id), float(row.clicks + 0.1 * row.impressions))
        for row in popularity.itertuples(index=False)
    ]


@lru_cache(maxsize=8)
def _load_seen_banner_history(interactions_csv: str) -> dict[str, set[str]]:
    interactions = pd.read_csv(interactions_csv, usecols=["user_id", "banner_id"])
    grouped = interactions.groupby("user_id")["banner_id"].agg(list)
    return {
        str(user_id): {str(banner_id) for banner_id in banner_ids}
        for user_id, banner_ids in grouped.items()
    }


class RetrievalService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.default_artifact_dir = _resolve_path(
            settings.project_root,
            None,
            DEFAULT_RETRIEVAL_ARTIFACTS,
        )
        self.default_banners_csv = _resolve_path(
            settings.project_root,
            None,
            DEFAULT_BANNERS,
        )
        self.default_interactions_csv = _resolve_path(
            settings.project_root,
            None,
            DEFAULT_INTERACTIONS,
        )
        self.state: RetrievalState | None = None
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

    def load(self) -> None:
        started_at = perf_counter()
        runtime = self._get_runtime(None)
        self.refresh_state()
        logger.info(
            "retrieval runtime loaded model_version=%s artifact_dir=%s num_users=%s num_items=%s load_ms=%.2f",
            runtime.model_version,
            runtime.artifact_dir,
            len(runtime.user2idx),
            len(runtime.item2idx),
            (perf_counter() - started_at) * 1000,
        )

    def refresh_state(self) -> RetrievalState:
        active_banner_ids: set[str] = set()
        popular_banner_scores: list[tuple[str, float]] = []
        seen_banner_history: dict[str, set[str]] = {}

        if self.default_banners_csv.exists():
            active_banner_ids = _load_active_banner_ids(str(self.default_banners_csv))
        if self.default_interactions_csv.exists():
            popular_banner_scores = _load_popular_banner_scores(str(self.default_interactions_csv))
            seen_banner_history = _load_seen_banner_history(str(self.default_interactions_csv))

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
        self._write_seen_banner_ids_to_redis(user_id, current_seen)
        logger.info(
            "retrieval seen updated user_id=%s added_count=%s total_seen_count=%s redis_enabled=%s",
            user_id,
            len(banner_ids),
            len(current_seen),
            self._redis_available(),
        )

    def _get_runtime(self, artifacts_dir: str | None) -> RetrievalRuntime:
        artifact_path = _resolve_path(
            self.settings.project_root,
            artifacts_dir,
            DEFAULT_RETRIEVAL_ARTIFACTS,
        )
        return _load_runtime(str(artifact_path))

    def _get_seen_items(
        self,
        request: RetrievalRequest,
        runtime: RetrievalRuntime,
    ) -> set[int]:
        if not request.exclude_seen:
            return set()

        interactions_csv = _resolve_path(
            self.settings.project_root,
            request.interactions_csv,
            DEFAULT_INTERACTIONS,
        )
        if interactions_csv == self.default_interactions_csv:
            seen_banner_ids = self._read_seen_banner_ids_from_redis(request.user_id)
            if seen_banner_ids is None:
                seen_banner_ids = self._get_state().seen_banner_history.get(request.user_id, set())
        else:
            seen_history = _load_seen_banner_history(str(interactions_csv))
            seen_banner_ids = seen_history.get(request.user_id, set())
        return {
            runtime.item2idx[banner_id]
            for banner_id in seen_banner_ids
            if banner_id in runtime.item2idx
        }

    def _get_active_banner_ids(self, request: RetrievalRequest) -> set[str] | None:
        if not request.only_active:
            return None

        banners_csv = _resolve_path(
            self.settings.project_root,
            request.banners_csv,
            DEFAULT_BANNERS,
        )
        if banners_csv == self.default_banners_csv:
            active_banner_ids = self._read_active_banner_ids_from_redis()
            if active_banner_ids is not None and active_banner_ids:
                return active_banner_ids
            return self._get_state().active_banner_ids
        return _load_active_banner_ids(str(banners_csv))

    def _search_top_k(
        self,
        runtime: RetrievalRuntime,
        user_id: str,
        top_k: int,
        seen_items: set[int],
        active_banner_ids: set[str] | None,
    ) -> list[RetrievalCandidate]:
        user_idx = runtime.user2idx[user_id]
        with torch.no_grad():
            user_tensor = torch.tensor([user_idx], dtype=torch.long, device=runtime.device)
            user_vector = runtime.model.encode_user(user_tensor)
            scores = torch.matmul(user_vector, runtime.item_embeddings.T).squeeze(0).clone()

            if seen_items:
                seen_tensor = torch.tensor(sorted(seen_items), dtype=torch.long, device=runtime.device)
                scores[seen_tensor] = -torch.inf

            if active_banner_ids is not None:
                inactive_indices = [
                    item_idx
                    for item_idx, banner_id in runtime.idx2item.items()
                    if banner_id not in active_banner_ids
                ]
                if inactive_indices:
                    inactive_tensor = torch.tensor(
                        inactive_indices,
                        dtype=torch.long,
                        device=runtime.device,
                    )
                    scores[inactive_tensor] = -torch.inf

            k = min(top_k, scores.numel())
            top_scores, top_indices = torch.topk(scores, k=k)

        items: list[RetrievalCandidate] = []
        for rank, (item_idx, score) in enumerate(
            zip(top_indices.cpu().tolist(), top_scores.cpu().tolist(), strict=False),
            start=1,
        ):
            if not np.isfinite(score):
                continue
            items.append(
                RetrievalCandidate(
                    banner_id=runtime.idx2item[item_idx],
                    retrieval_rank=rank,
                    retrieval_score=float(score),
                )
            )
        return items

    def _fallback_candidates(
        self,
        request: RetrievalRequest,
        active_banner_ids: set[str] | None,
    ) -> list[RetrievalCandidate]:
        interactions_csv = _resolve_path(
            self.settings.project_root,
            request.interactions_csv,
            DEFAULT_INTERACTIONS,
        )
        if interactions_csv == self.default_interactions_csv:
            popular = self._read_popular_banner_scores_from_redis()
            if popular is None or not popular:
                popular = self._get_state().popular_banner_scores
        else:
            popular = _load_popular_banner_scores(str(interactions_csv))
        items: list[RetrievalCandidate] = []
        for banner_id, score in popular:
            if active_banner_ids is not None and banner_id not in active_banner_ids:
                continue
            items.append(
                RetrievalCandidate(
                    banner_id=banner_id,
                    retrieval_rank=len(items) + 1,
                    retrieval_score=score,
                )
            )
            if len(items) >= request.top_k:
                break
        return items

    def get_candidates(self, request: RetrievalRequest) -> RetrievalResult:
        started_at = perf_counter()
        runtime = self._get_runtime(request.artifacts_dir)
        active_banner_ids = self._get_active_banner_ids(request)

        if request.user_id in runtime.user2idx:
            seen_items = self._get_seen_items(request, runtime)
            items = self._search_top_k(
                runtime=runtime,
                user_id=request.user_id,
                top_k=request.top_k,
                seen_items=seen_items,
                active_banner_ids=active_banner_ids,
            )
            source = "two_tower"
        else:
            items = self._fallback_candidates(request, active_banner_ids)
            source = "popular_fallback"

        if not items:
            raise InvalidRequestError("No retrieval candidates available for this request.")

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
        state = self.refresh_state()
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
        _load_runtime.cache_clear()
        runtime = self._get_runtime(None)
        state = self.refresh_state()
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
