from __future__ import annotations

import unittest

import pandas as pd

from backend.src.core.config import get_settings
from backend.src.schemas.recommendations import RecommendationRequest
from backend.src.schemas.retrieval import RetrievalRequest
from backend.src.services.recommendations import recommend_banners
from backend.src.services.retrieval import RetrievalService, retrieve_banners


class RecommendationsWithRetrievalTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.settings = get_settings()
        cls.retrieval_service = RetrievalService(cls.settings)
        cls.retrieval_service.load()
        interactions = pd.read_csv("data/db/banner_interactions.csv")
        cls.original_seen_banner_ids = set(
            interactions.loc[interactions["user_id"] == "u_00007", "banner_id"].astype(str)
        )

    def test_recommendations_use_retrieval_candidates(self) -> None:
        response = recommend_banners(
            RecommendationRequest(
                user_id="u_00007",
                top_k=3,
                retrieval_artifacts_dir="artifacts/pytorch_retrieval",
            ),
            self.settings,
            self.retrieval_service,
        )

        self.assertEqual(response.user_id, "u_00007")
        self.assertEqual(response.candidate_mode, "retrieval + ranking")
        self.assertEqual(response.model_type, "deepfm")
        self.assertEqual(len(response.items), 3)
        self.assertIsNotNone(response.items[0].retrieval_rank)
        self.assertIsNotNone(response.items[0].retrieval_score)

    def test_recommendations_update_live_seen_history_for_retrieval(self) -> None:
        local_service = RetrievalService(self.settings)
        local_service.load()

        response = recommend_banners(
            RecommendationRequest(
                user_id="u_00007",
                top_k=3,
                retrieval_artifacts_dir="artifacts/pytorch_retrieval",
            ),
            self.settings,
            local_service,
        )

        newly_served_banner_ids = {
            item.banner_id
            for item in response.items
            if item.banner_id not in self.original_seen_banner_ids
        }
        self.assertTrue(newly_served_banner_ids)

        retrieval_response = retrieve_banners(
            RetrievalRequest(user_id="u_00007", top_k=20, exclude_seen=True),
            local_service,
        )
        filtered_banner_ids = {item.banner_id for item in retrieval_response.items}
        self.assertTrue(filtered_banner_ids.isdisjoint(newly_served_banner_ids))


if __name__ == "__main__":
    unittest.main()
