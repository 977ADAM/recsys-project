from __future__ import annotations

import unittest

import pandas as pd
import torch

from src.retrieval.data.preprocess import build_negatives_pairs
from src.retrieval.models.two_tower import TwoTower


class RetrievalModelingTestCase(unittest.TestCase):
    def test_build_negatives_pairs_prefers_user_hard_negatives(self) -> None:
        frame = pd.DataFrame(
            [
                {"user_id": "u1", "banner_id": "b1", "impressions": 5, "clicks": 1},
                {"user_id": "u1", "banner_id": "b2", "impressions": 7, "clicks": 0},
                {"user_id": "u1", "banner_id": "b3", "impressions": 3, "clicks": 0},
                {"user_id": "u2", "banner_id": "b4", "impressions": 10, "clicks": 4},
                {"user_id": "u3", "banner_id": "b5", "impressions": 9, "clicks": 2},
            ]
        )

        pairs = build_negatives_pairs(
            frame,
            negatives_per_positive=2,
            hard_negative_ratio=0.5,
            seed=42,
        )

        u1_negatives = set(
            pairs.loc[(pairs["user_id"] == "u1") & (pairs["label"] == 0.0), "banner_id"].tolist()
        )
        self.assertIn("b2", u1_negatives)

    def test_two_tower_mlp_keeps_embedding_shape(self) -> None:
        model = TwoTower(
            n_users=3,
            n_banners=4,
            emb_dim=16,
            hidden_dims=(32, 24),
            dropout=0.1,
        )

        user_ids = torch.tensor([0, 1], dtype=torch.long)
        banner_ids = torch.tensor([2, 3], dtype=torch.long)

        user_vectors = model.encode_user(user_ids)
        banner_vectors = model.encode_banner(banner_ids)
        all_scores = model.score_all_banners()

        self.assertEqual(tuple(user_vectors.shape), (2, 16))
        self.assertEqual(tuple(banner_vectors.shape), (2, 16))
        self.assertEqual(tuple(all_scores.shape), (3, 4))


if __name__ == "__main__":
    unittest.main()
