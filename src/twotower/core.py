from dataclasses import dataclass
from pathlib import Path

import torch

from src.checkpoint import (
    build_run_summary,
    load_checkpoint,
    save_checkpoint,
    save_run_summary,
)
from src.config import TwoTowerConfig
from src.data import EvalData, TrainData
from src.encoding import EncodedTable
from src.model import FitResult, TwoTowerModel, init_model


@dataclass
class TwoTowerArtifacts:
    fit_result: FitResult | None = None
    train_data: TrainData | None = None
    valid_data: EvalData | None = None
    last_eval_data: EvalData | None = None
    metrics: dict[str, float] | None = None
    summary_path: Path | None = None
    checkpoint_path: Path | None = None


class TwoTower:
    def __init__(self, config: TwoTowerConfig | None = None):
        self.config = config if config is not None else TwoTowerConfig()
        self.device = torch.device(self.config.device)
        self.model: TwoTowerModel | None = None
        self.artifacts = TwoTowerArtifacts()
        self.is_fitted = False

    def fit(
        self,
        train_data: TrainData,
        valid_data: EvalData,
    ) -> "TwoTower":
        train_split = train_data
        valid_split = self._require_valid_data(valid_data)
        model_args = self.config.to_namespace()

        model = init_model(model_args, train_split.user_table, train_split.item_table, self.device)
        fit_result = model.fit(
            args=model_args,
            train_loader=train_split.train_loader,
            valid_pairs=valid_split.pairs,
            user_table=train_split.user_table,
            item_table=train_split.item_table,
            device=self.device,
        )

        self.model = model
        self.artifacts.fit_result = fit_result
        self.artifacts.train_data = train_split
        self.artifacts.valid_data = valid_split
        self.is_fitted = True
        return self

    def evaluate(
        self,
        test_data: EvalData,
    ) -> dict[str, float]:
        self._ensure_ready()
        eval_data = test_data
        recall, eval_users = self.model.evaluate_recall_at_k(
            user_table=eval_data.user_table,
            item_table=eval_data.item_table,
            positives=eval_data.pairs,
            k=self.config.recall_k,
            device=self.device,
            batch_size=self.config.eval_batch_size,
        )
        metrics = {
            "best_epoch": float(self.artifacts.fit_result.best_epoch),
            "valid_recall_at_k": self.artifacts.fit_result.best_valid_recall_at_k,
            "valid_users": float(self.artifacts.fit_result.best_valid_users),
            f"{eval_data.split_name}_recall_at_k": recall,
            f"{eval_data.split_name}_users": float(eval_users),
        }
        self.artifacts.last_eval_data = eval_data
        self.artifacts.metrics = metrics
        return metrics

    def predict(
        self,
        user_ids: list[str],
        k: int | None = None,
        batch_size: int | None = None,
    ) -> list[dict]:
        self._ensure_ready()
        user_table, item_table = self._require_tables()
        unknown_user_ids = [user_id for user_id in user_ids if user_id not in user_table.id_to_row]
        if unknown_user_ids:
            missing = ", ".join(unknown_user_ids[:3])
            suffix = "" if len(unknown_user_ids) <= 3 else ", ..."
            raise ValueError(f"Unknown user_id values: {missing}{suffix}")

        user_indices = [user_table.id_to_row[user_id] for user_id in user_ids]
        scores, item_indices = self.predict_indices(
            user_indices=user_indices,
            k=k,
            batch_size=batch_size,
        )

        predictions: list[dict] = []
        for row_idx, user_id in enumerate(user_ids):
            items = [
                {
                    "banner_id": item_table.ids[item_index],
                    "score": float(score),
                }
                for score, item_index in zip(scores[row_idx].tolist(), item_indices[row_idx].tolist())
            ]
            predictions.append(
                {
                    "user_id": user_id,
                    "items": items,
                }
            )
        return predictions

    def predict_indices(
        self,
        user_indices,
        k: int | None = None,
        batch_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_ready()
        user_table, item_table = self._require_tables()
        self.model.before_evaluate(
            user_table=user_table,
            item_table=item_table,
            device=self.device,
            test_batch_size=batch_size or self.config.eval_batch_size,
        )
        return self.model.predict(
            eval_users=user_indices,
            test_batch_size=batch_size or self.config.eval_batch_size,
            k=k or self.config.recall_k,
        )

    def save_model(self, output_path: Path | str | None = None) -> Path:
        self._ensure_ready()
        metrics = self._require_metrics()
        user_table, item_table = self._require_tables()
        checkpoint_path = Path(output_path) if output_path is not None else Path(self.config.output)
        split_dates = self._split_dates()
        save_checkpoint(
            output_path=checkpoint_path,
            model=self.model,
            config=self.config,
            metrics=metrics,
            history=self.artifacts.fit_result.history,
            split_dates=split_dates,
            user_table=user_table,
            item_table=item_table,
        )
        self.artifacts.checkpoint_path = checkpoint_path
        return checkpoint_path

    def save_summary(self, output_path: Path | str | None = None) -> Path:
        self._ensure_ready()
        metrics = self._require_metrics()
        summary_path = (
            Path(output_path)
            if output_path is not None
            else Path(self.config.output).with_name(f"{Path(self.config.output).stem}_summary.json")
        )
        eval_split_name = self._current_eval_split_name()
        run_summary = build_run_summary(
            config=self.config,
            device=self.device,
            data=self._summary_dataset_stats(),
            fit_result=self.artifacts.fit_result,
            test_recall=metrics[f"{eval_split_name}_recall_at_k"],
            test_users=int(metrics[f"{eval_split_name}_users"]),
        )
        save_run_summary(summary_path, run_summary)
        self.artifacts.summary_path = summary_path
        return summary_path

    @classmethod
    def load_model(cls, checkpoint_path: Path | str, device: str = "cpu") -> "TwoTower":
        checkpoint = load_checkpoint(Path(checkpoint_path), map_location=device)
        config_dict = checkpoint["config"]
        config_dict["device"] = device
        instance = cls(config=TwoTowerConfig.from_dict(config_dict))
        user_table = checkpoint["user_table"]
        item_table = checkpoint["item_table"]
        model = init_model(
            args=instance.config.to_namespace(),
            user_table=user_table,
            item_table=item_table,
            device=instance.device,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        instance.model = model
        instance.artifacts.fit_result = FitResult(
            history=checkpoint.get("history", []),
            best_epoch=int(checkpoint["metrics"].get("best_epoch", 0)),
            best_valid_recall_at_k=float(checkpoint["metrics"].get("valid_recall_at_k", float("nan"))),
            best_valid_users=int(checkpoint["metrics"].get("valid_users", 0)),
        )
        instance.artifacts.train_data = TrainData(
            user_table=user_table,
            item_table=item_table,
            train_loader=None,
            train_pairs=None,
        )
        instance.artifacts.metrics = checkpoint.get("metrics")
        instance.is_fitted = True
        return instance

    def _split_dates(self) -> dict[str, str]:
        return {
            "train_end": self.config.train_end,
            "valid_end": self.config.valid_end,
        }

    def _current_eval_split_name(self) -> str:
        if self.artifacts.last_eval_data is not None:
            return self.artifacts.last_eval_data.split_name
        return "test"

    def _ensure_ready(self) -> None:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Call fit() or load_model() before using the model.")

    def _require_metrics(self) -> dict[str, float]:
        if self.artifacts.metrics is None:
            raise RuntimeError("Metrics are unavailable. Call evaluate() before saving artifacts.")
        return self.artifacts.metrics

    def _require_tables(self) -> tuple[EncodedTable, EncodedTable]:
        train_data = self.artifacts.train_data
        if train_data is None:
            raise RuntimeError("Encoded tables are unavailable for the current model state.")
        return train_data.user_table, train_data.item_table

    def _require_valid_data(self, valid_data: EvalData | None) -> EvalData:
        if valid_data is None:
            raise ValueError("valid_data is required when calling fit(train_data, valid_data).")
        return valid_data

    def _summary_dataset_stats(self) -> dict[str, int]:
        train_data = self.artifacts.train_data
        valid_data = self.artifacts.valid_data
        test_data = self.artifacts.last_eval_data
        if train_data is None or valid_data is None or test_data is None:
            raise RuntimeError("Not enough split data to build summary. Call fit(...) and evaluate(...) first.")

        return {
            "num_users": len(train_data.user_table.ids),
            "num_items": len(train_data.item_table.ids),
            "num_train_pairs": 0 if train_data.train_pairs is None else len(train_data.train_pairs),
            "num_valid_pairs": len(valid_data.pairs),
            "num_test_pairs": len(test_data.pairs),
        }
