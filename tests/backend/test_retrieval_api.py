from __future__ import annotations

import unittest
from unittest.mock import patch

from backend.src.api.deps import get_retrieval_service
from backend.src.api.v1.retrieval import (
    get_retrieval_candidates,
    refresh_retrieval,
    reload_retrieval,
)
from backend.src.schemas.retrieval import RetrievalRequest


async def _run_directly(func, *args, **kwargs):
    return func(*args, **kwargs)


class RetrievalApiTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_handler_returns_known_user_payload(self) -> None:
        with patch("backend.src.api.v1.retrieval.run_in_threadpool", new=_run_directly):
            response = await get_retrieval_candidates(
                RetrievalRequest(user_id="u_00007", top_k=3),
                get_retrieval_service(),
            )

        self.assertEqual(response.user_id, "u_00007")
        self.assertEqual(response.source, "two_tower")
        self.assertEqual(len(response.items), 3)
        self.assertEqual(response.items[0].retrieval_rank, 1)

    async def test_refresh_handler_returns_state_stats(self) -> None:
        with patch("backend.src.api.v1.retrieval.run_in_threadpool", new=_run_directly):
            response = await refresh_retrieval(get_retrieval_service())

        self.assertEqual(response.model_version, "pytorch_retrieval")
        self.assertGreater(response.active_banner_count, 0)
        self.assertGreater(response.popular_banner_count, 0)
        self.assertGreater(response.seen_user_count, 0)

    async def test_reload_handler_returns_runtime_stats(self) -> None:
        with patch("backend.src.api.v1.retrieval.run_in_threadpool", new=_run_directly):
            response = await reload_retrieval(get_retrieval_service())

        self.assertEqual(response.model_version, "pytorch_retrieval")
        self.assertEqual(response.embedding_dim, 64)
        self.assertGreater(response.num_users, 0)
        self.assertGreater(response.num_items, 0)
        self.assertGreater(response.active_banner_count, 0)


if __name__ == "__main__":
    unittest.main()
