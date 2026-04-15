from __future__ import annotations

import unittest

import torch

from src.retrieval.models.train import train_model
from src.retrieval.models.two_tower import TwoTower


class TrainModelEarlyStoppingTestCase(unittest.TestCase):
    def test_train_model_stops_when_callback_requests_it(self) -> None:
        model = TwoTower(n_users=3, n_banners=4, emb_dim=8)
        users = torch.tensor([0, 1, 2, 0], dtype=torch.long)
        banners = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)

        visited_epochs: list[int] = []

        def stop_after_second_epoch(_: TwoTower, epoch: int, loss: float) -> bool:
            self.assertGreaterEqual(loss, 0.0)
            visited_epochs.append(epoch)
            return epoch >= 2

        train_model(
            model,
            users,
            banners,
            labels,
            epochs=10,
            lr=0.01,
            batch_size=2,
            shuffle=False,
            epoch_callback=stop_after_second_epoch,
        )

        self.assertEqual(visited_epochs, [1, 2])


if __name__ == "__main__":
    unittest.main()
