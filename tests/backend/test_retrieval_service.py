from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

import pandas as pd
import torch

from backend.src.core.config import Settings, get_settings
from backend.src.schemas.retrieval import RetrievalRequest
from backend.src.services.retrieval import RetrievalService, retrieve_banners
from src.retrieval.models.two_tower import TwoTower

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DATA_DIR = PROJECT_ROOT / "data" / "db"
FIXTURE_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "pytorch_retrieval"


def _build_test_settings(project_root: Path) -> Settings:
    return Settings(
        project_root=project_root,
        app_name="Recsys API Test",
        api_v1_prefix="/api/v1",
        database_url="postgresql+psycopg://recsys:recsys@localhost:5432/recsys",
        redis_url="redis://127.0.0.1:6399/0",
    )


def _copy_default_retrieval_fixture(project_root: Path) -> None:
    data_dir = project_root / "data" / "db"
    artifacts_dir = project_root / "artifacts" / "pytorch_retrieval"
    data_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(FIXTURE_DATA_DIR / "banner_interactions.csv", data_dir / "banner_interactions.csv")
    shutil.copy2(FIXTURE_DATA_DIR / "banners.csv", data_dir / "banners.csv")
    shutil.copytree(FIXTURE_ARTIFACTS_DIR, artifacts_dir, dirs_exist_ok=True)


def _write_runtime_fixture(
    artifact_dir: Path,
    *,
    user_id: str,
    banner_id: str,
    embedding_dim: int,
    model_version: str,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model = TwoTower(n_users=1, n_banners=1, emb_dim=embedding_dim)
    with torch.no_grad():
        model.user_tower.weight.fill_(1.0)
        model.banner_tower.weight.fill_(1.0)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "embedding_dim": embedding_dim,
            "n_users": 1,
            "n_banners": 1,
        },
        artifact_dir / "model.pt",
    )

    with (artifact_dir / "mappings.json").open("w", encoding="utf-8") as file_obj:
        json.dump(
            {
                "user2idx": {user_id: 0},
                "item2idx": {banner_id: 0},
                "idx2item": {"0": banner_id},
            },
            file_obj,
            ensure_ascii=False,
            indent=2,
        )

    with (artifact_dir / "metadata.json").open("w", encoding="utf-8") as file_obj:
        json.dump(
            {
                "model_version": model_version,
                "model_type": "two_tower",
                "embedding_dim": embedding_dim,
            },
            file_obj,
            ensure_ascii=False,
            indent=2,
        )


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
        self.assertTrue(response.items[0].banner_id.startswith("b_"))
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

class RetrievalRefreshReloadTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_root = Path(self.temp_dir.name)
        _copy_default_retrieval_fixture(self.project_root)
        self.service = RetrievalService(_build_test_settings(self.project_root))
        self.service.load()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_refresh_picks_up_updated_default_csvs(self) -> None:
        banners_path = self.project_root / "data" / "db" / "banners.csv"
        interactions_path = self.project_root / "data" / "db" / "banner_interactions.csv"

        pd.DataFrame(
            [
                {"banner_id": "b_state", "is_active": 1},
                {"banner_id": "b_inactive", "is_active": 0},
            ]
        ).to_csv(banners_path, index=False)
        pd.DataFrame(
            [
                {"user_id": "u_state", "banner_id": "b_state", "impressions": 10, "clicks": 3},
                {"user_id": "u_state", "banner_id": "b_inactive", "impressions": 5, "clicks": 0},
            ]
        ).to_csv(interactions_path, index=False)

        refreshed = self.service.refresh()
        response = retrieve_banners(
            RetrievalRequest(user_id="unknown_user", top_k=5, only_active=True),
            self.service,
        )

        self.assertEqual(refreshed.active_banner_count, 1)
        self.assertEqual(refreshed.popular_banner_count, 2)
        self.assertEqual(refreshed.seen_user_count, 1)
        self.assertEqual([item.banner_id for item in response.items], ["b_state"])

    def test_refresh_preserves_live_seen_history(self) -> None:
        interactions = pd.read_csv(self.project_root / "data" / "db" / "banner_interactions.csv")
        original_seen_banner_ids = set(
            interactions.loc[interactions["user_id"] == "u_00007", "banner_id"].astype(str)
        )

        initial_response = retrieve_banners(
            RetrievalRequest(user_id="u_00007", top_k=20),
            self.service,
        )
        newly_served_banner_id = next(
            (
                item.banner_id
                for item in initial_response.items
                if item.banner_id not in original_seen_banner_ids
            ),
            None,
        )
        self.assertIsNotNone(newly_served_banner_id)

        self.service.record_served_banners("u_00007", [newly_served_banner_id])
        self.service.refresh()

        filtered_response = retrieve_banners(
            RetrievalRequest(user_id="u_00007", top_k=50, exclude_seen=True),
            self.service,
        )
        filtered_banner_ids = {item.banner_id for item in filtered_response.items}

        self.assertNotIn(newly_served_banner_id, filtered_banner_ids)

    def test_reload_picks_up_replaced_runtime_artifacts(self) -> None:
        artifact_dir = self.project_root / "artifacts" / "pytorch_retrieval"
        _write_runtime_fixture(
            artifact_dir,
            user_id="u_reload",
            banner_id="b_reload",
            embedding_dim=8,
            model_version="reloaded_fixture",
        )

        reloaded = self.service.reload()
        response = retrieve_banners(
            RetrievalRequest(user_id="u_reload", top_k=1),
            self.service,
        )

        self.assertEqual(reloaded.model_version, "reloaded_fixture")
        self.assertEqual(reloaded.embedding_dim, 8)
        self.assertEqual(reloaded.num_users, 1)
        self.assertEqual(reloaded.num_items, 1)
        self.assertEqual([item.banner_id for item in response.items], ["b_reload"])


if __name__ == "__main__":
    unittest.main()
