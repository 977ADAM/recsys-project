import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import EncodedTable


@dataclass
class FitResult:
    history: list[dict[str, float]]
    best_epoch: int
    best_valid_recall_at_k: float
    best_valid_users: int

class BaseModel(nn.Module):
    def __init__(self, temperature: float = 0.05) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.temperature = float(temperature)
        self._cached_user_embeddings: torch.Tensor | None = None
        self._cached_item_embeddings: torch.Tensor | None = None

    def _clear_eval_cache(self) -> None:
        self._cached_user_embeddings = None
        self._cached_item_embeddings = None

    def _clone_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu().clone()
            for key, value in self.state_dict().items()
        }

    def encode_users(self, categorical: torch.Tensor, numerical: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def encode_items(self, categorical: torch.Tensor, numerical: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        user_categorical: torch.Tensor,
        user_numerical: torch.Tensor,
        item_categorical: torch.Tensor,
        item_numerical: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        user_embeddings = self.encode_users(user_categorical, user_numerical)
        item_embeddings = self.encode_items(item_categorical, item_numerical)
        logits = user_embeddings @ item_embeddings.T / self.temperature
        return user_embeddings, item_embeddings, logits

    def train_one_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        train_loader,
        user_table: EncodedTable,
        item_table: EncodedTable,
        device: torch.device,
    ) -> float:
        self.train()
        self._clear_eval_cache()
        total_loss = 0.0
        total_batches = 0

        for user_idx, item_idx, weights in train_loader:
            user_idx = user_idx.to(dtype=torch.long)
            item_idx = item_idx.to(dtype=torch.long)
            weights = weights.to(device)

            user_categorical = user_table.categorical[user_idx].to(device)
            user_numerical = user_table.numerical[user_idx].to(device)
            item_categorical = item_table.categorical[item_idx].to(device)
            item_numerical = item_table.numerical[item_idx].to(device)

            _, _, logits = self(
                user_categorical,
                user_numerical,
                item_categorical,
                item_numerical,
            )
            targets = torch.arange(logits.shape[0], device=device)

            loss_users = F.cross_entropy(logits, targets, reduction="none")
            loss_items = F.cross_entropy(logits.T, targets, reduction="none")
            sample_weights = weights / weights.mean().clamp(min=1e-6)
            loss = (((loss_users + loss_items) * 0.5) * sample_weights).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_batches += 1

        return total_loss / max(total_batches, 1)

    @torch.inference_mode()
    def _encode_table(
        self,
        table: EncodedTable,
        tower: str,
        device: torch.device,
        batch_size: int,
    ) -> torch.Tensor:
        encoded_batches: list[torch.Tensor] = []
        effective_batch_size = max(int(batch_size), 1)
        for start in range(0, len(table.ids), effective_batch_size):
            stop = start + effective_batch_size
            categorical = table.categorical[start:stop].to(device)
            numerical = table.numerical[start:stop].to(device)
            if tower == "user":
                batch_embeddings = self.encode_users(categorical, numerical)
            elif tower == "item":
                batch_embeddings = self.encode_items(categorical, numerical)
            else:
                raise ValueError(f"Unknown tower '{tower}'. Expected 'user' or 'item'.")
            encoded_batches.append(batch_embeddings.cpu())
        return torch.cat(encoded_batches, dim=0)

    @torch.inference_mode()
    def before_evaluate(
        self,
        user_table: EncodedTable,
        item_table: EncodedTable,
        device: torch.device,
        test_batch_size: int,
    ) -> None:
        self.eval()
        self._cached_user_embeddings = self._encode_table(
            table=user_table,
            tower="user",
            device=device,
            batch_size=test_batch_size,
        )
        self._cached_item_embeddings = self._encode_table(
            table=item_table,
            tower="item",
            device=device,
            batch_size=test_batch_size,
        )

    @torch.inference_mode()
    def predict(
        self,
        eval_users: torch.Tensor,
        eval_pos: pd.DataFrame | None = None,
        test_batch_size: int = 4096,
        k: int = 100,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del eval_pos
        if self._cached_user_embeddings is None or self._cached_item_embeddings is None:
            raise RuntimeError("Call before_evaluate() before predict().")

        if not torch.is_tensor(eval_users):
            eval_users = torch.tensor(list(eval_users), dtype=torch.long)

        eval_users = eval_users.to(dtype=torch.long).cpu()
        candidate_count = int(self._cached_item_embeddings.shape[0])
        top_k = min(k, candidate_count)
        if eval_users.numel() == 0:
            empty_scores = torch.empty((0, top_k), dtype=torch.float32)
            empty_indices = torch.empty((0, top_k), dtype=torch.long)
            return empty_scores, empty_indices

        effective_batch_size = max(int(test_batch_size), 1)
        score_batches: list[torch.Tensor] = []
        index_batches: list[torch.Tensor] = []
        for start in range(0, eval_users.numel(), effective_batch_size):
            batch_users = eval_users[start : start + effective_batch_size]
            batch_user_embeddings = self._cached_user_embeddings.index_select(0, batch_users)
            batch_scores, batch_top_indices = torch.topk(
                batch_user_embeddings @ self._cached_item_embeddings.T,
                k=top_k,
                dim=1,
            )
            score_batches.append(batch_scores)
            index_batches.append(batch_top_indices)

        return torch.cat(score_batches, dim=0), torch.cat(index_batches, dim=0)

    @torch.inference_mode()
    def evaluate_recall_at_k(
        self,
        user_table: EncodedTable,
        item_table: EncodedTable,
        positives: pd.DataFrame,
        k: int,
        device: torch.device,
        batch_size: int,
    ) -> tuple[float, int]:
        relevant_by_user: dict[int, set[int]] = {}
        for pair in positives.itertuples():
            user_row = user_table.id_to_row.get(str(pair.user_id))
            item_row = item_table.id_to_row.get(str(pair.banner_id))
            if user_row is None or item_row is None:
                continue
            relevant_by_user.setdefault(user_row, set()).add(item_row)

        if not relevant_by_user:
            return float("nan"), 0

        self.before_evaluate(
            user_table=user_table,
            item_table=item_table,
            device=device,
            test_batch_size=batch_size,
        )
        relevant_user_rows = torch.tensor(sorted(relevant_by_user), dtype=torch.long)
        batch_scores, batch_top_indices = self.predict(
            eval_users=relevant_user_rows,
            test_batch_size=batch_size,
            k=k,
        )

        recall_sum = 0.0
        for offset, user_row in enumerate(relevant_user_rows.tolist()):
            relevant_items = relevant_by_user[user_row]
            predicted_scores = batch_scores[offset].tolist()
            predicted_items = batch_top_indices[offset].tolist()
            hits = sum(
                1
                for score, item_idx in zip(predicted_scores, predicted_items)
                if score > 0 and item_idx in relevant_items
            )
            recall_sum += hits / len(relevant_items)

        return recall_sum / len(relevant_user_rows), len(relevant_user_rows)

    def fit(
        self,
        args: argparse.Namespace,
        train_loader,
        valid_pairs: pd.DataFrame,
        user_table: EncodedTable,
        item_table: EncodedTable,
        device: torch.device,
    ) -> FitResult:
        optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)
        history: list[dict[str, float]] = []
        best_state_dict = self._clone_state_dict()
        best_epoch = 0
        best_valid_recall = float("-inf")
        best_valid_users = 0
        epochs_without_improvement = 0
        patience = max(int(args.patience), 0)

        print("Training Two-Tower model...")
        for epoch in range(1, args.epochs + 1):
            train_loss = self.train_one_epoch(
                optimizer=optimizer,
                train_loader=train_loader,
                user_table=user_table,
                item_table=item_table,
                device=device,
            )
            valid_recall, valid_users = self.evaluate_recall_at_k(
                user_table=user_table,
                item_table=item_table,
                positives=valid_pairs,
                k=args.recall_k,
                device=device,
                batch_size=args.eval_batch_size,
            )
            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": train_loss,
                    "valid_recall_at_k": valid_recall,
                }
            )
            print(
                f"epoch={epoch:02d}",
                f"train_loss={train_loss:.4f}",
                f"valid_recall@{args.recall_k}={valid_recall:.4f}",
                f"valid_users={valid_users}",
            )

            candidate_score = valid_recall if np.isfinite(valid_recall) else float("-inf")
            improved = epoch == 1 or candidate_score > best_valid_recall
            if improved:
                best_state_dict = self._clone_state_dict()
                best_epoch = epoch
                best_valid_recall = candidate_score
                best_valid_users = valid_users
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if patience and epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch:02d} with patience={patience}")
                break

        self.load_state_dict(best_state_dict)
        self._clear_eval_cache()
        restored_recall = float("nan") if best_valid_recall == float("-inf") else best_valid_recall
        print(
            "Training done.",
            f"best_epoch={best_epoch}",
            f"best_valid_recall@{args.recall_k}={restored_recall:.4f}",
            f"best_valid_users={best_valid_users}",
        )
        return FitResult(
            history=history,
            best_epoch=best_epoch,
            best_valid_recall_at_k=restored_recall,
            best_valid_users=best_valid_users,
        )
    


class Tower(nn.Module):
    def __init__(
        self,
        cardinalities: list[int],
        num_numeric_features: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            nn.Embedding(num_embeddings=cardinality, embedding_dim=embedding_dim)
            for cardinality in cardinalities
        )
        input_dim = len(cardinalities) * embedding_dim + num_numeric_features
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, categorical: torch.Tensor, numerical: torch.Tensor) -> torch.Tensor:
        parts = [embedding(categorical[:, idx]) for idx, embedding in enumerate(self.embeddings)]
        if numerical.shape[1] > 0:
            parts.append(numerical)
        tower_input = torch.cat(parts, dim=1)
        return F.normalize(self.network(tower_input), dim=1)
    


class TwoTowerModel(BaseModel):
    def __init__(
        self,
        user_cardinalities: list[int],
        item_cardinalities: list[int],
        user_num_features: int,
        item_num_features: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        temperature: float,
    ) -> None:
        super().__init__(temperature=temperature)
        self.user_tower = Tower(
            cardinalities=user_cardinalities,
            num_numeric_features=user_num_features,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )
        self.item_tower = Tower(
            cardinalities=item_cardinalities,
            num_numeric_features=item_num_features,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )

    def encode_users(self, categorical: torch.Tensor, numerical: torch.Tensor) -> torch.Tensor:
        return self.user_tower(categorical, numerical)

    def encode_items(self, categorical: torch.Tensor, numerical: torch.Tensor) -> torch.Tensor:
        return self.item_tower(categorical, numerical)
    



def init_model(
    args: argparse.Namespace,
    user_table: EncodedTable,
    item_table: EncodedTable,
    device: torch.device,
) -> TwoTowerModel:
    print("Initializing Two-Tower model...")
    model = TwoTowerModel(
        user_cardinalities=user_table.cardinalities,
        item_cardinalities=item_table.cardinalities,
        user_num_features=int(user_table.numerical.shape[1]),
        item_num_features=int(item_table.numerical.shape[1]),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        temperature=args.temperature,
    ).to(device)
    return model



def iterate_minibatches(num_examples: int, batch_size: int):
    order = torch.randperm(num_examples)
    for start in range(0, num_examples, batch_size):
        batch = order[start : start + batch_size]
        if batch.numel() > 1:
            yield batch