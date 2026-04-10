from __future__ import annotations

import unittest

import pandas as pd

from backend.src.core.config import get_settings
from backend.src.schemas.retrieval import RetrievalRequest
from backend.src.services.retrieval import RetrievalService, retrieve_banners


class RetrievalServiceTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = RetrievalService(get_settings())
        cls.service.load()
        interactions = pd.read_csv("data/db/banner_interactions.csv")
        banners = pd.read_csv("data/db/banners.csv")
        cls.seen_banner_ids = set(
            interactions.loc[interactions["user_id"] == "u_00007", "banner_id"].astype(str)
        )
        cls.inactive_banner_ids = set(
            banners.loc[banners["is_active"] == 0, "banner_id"].astype(str)
        )

    def test_returns_two_tower_candidates_for_known_user(self) -> None:
        response = retrieve_banners(
            RetrievalRequest(user_id="u_00007", top_k=5),
            self.service,
        )

        self.assertEqual(response.user_id, "u_00007")
        self.assertEqual(response.source, "two_tower")
        self.assertEqual(response.model_version, "pytorch_retrieval")
        self.assertEqual(len(response.items), 5)
        self.assertEqual(response.items[0].banner_id, "b_0106")
        self.assertEqual(response.items[0].retrieval_rank, 1)
        self.assertGreater(response.items[0].retrieval_score, 0.0)

    def test_returns_popular_fallback_for_unknown_user(self) -> None:
        response = retrieve_banners(
            RetrievalRequest(user_id="unknown_user", top_k=5, only_active=True),
            self.service,
        )

        self.assertEqual(response.user_id, "unknown_user")
        self.assertEqual(response.source, "popular_fallback")
        self.assertEqual(response.model_version, "pytorch_retrieval")
        self.assertEqual(len(response.items), 5)
        self.assertEqual(response.items[0].retrieval_rank, 1)
        self.assertGreater(response.items[0].retrieval_score, 0.0)

    def test_exclude_seen_removes_seen_banners_from_results(self) -> None:
        response = retrieve_banners(
            RetrievalRequest(user_id="u_00007", top_k=10, exclude_seen=True),
            self.service,
        )

        returned_banner_ids = {item.banner_id for item in response.items}
        self.assertTrue(returned_banner_ids)
        self.assertTrue(returned_banner_ids.isdisjoint(self.seen_banner_ids))

    def test_only_active_excludes_inactive_banners(self) -> None:
        response = retrieve_banners(
            RetrievalRequest(user_id="unknown_user", top_k=20, only_active=True),
            self.service,
        )

        returned_banner_ids = {item.banner_id for item in response.items}
        self.assertTrue(returned_banner_ids)
        self.assertTrue(returned_banner_ids.isdisjoint(self.inactive_banner_ids))


if __name__ == "__main__":
    unittest.main()
