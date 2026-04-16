from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.encoding import EncodedTable, encode_table
from src.features import (
    ITEM_CATEGORICAL_COLUMNS,
    ITEM_NUMERIC_COLUMNS,
    USER_CATEGORICAL_COLUMNS,
    USER_NUMERIC_COLUMNS,
)

@dataclass
class PairTensors:
    user_indices: torch.Tensor
    item_indices: torch.Tensor
    weights: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.user_indices.shape[0])


@dataclass
class DataBundle:
    user_table: EncodedTable
    item_table: EncodedTable
    train_loader: DataLoader
    train_pairs: pd.DataFrame
    valid_pairs: pd.DataFrame
    test_pairs: pd.DataFrame

    @property
    def num_users(self) -> int:
        return len(self.user_table.ids)

    @property
    def num_items(self) -> int:
        return len(self.item_table.ids)

    @property
    def num_train_pairs(self) -> int:
        return len(self.train_pairs)

    @property
    def num_valid_pairs(self) -> int:
        return len(self.valid_pairs)

    @property
    def num_test_pairs(self) -> int:
        return len(self.test_pairs)


class PairDataset(Dataset):
    def __init__(self, pairs: PairTensors):
        self.user_indices = pairs.user_indices
        self.item_indices = pairs.item_indices
        self.weights = pairs.weights

    def __len__(self) -> int:
        return int(self.user_indices.shape[0])

    def __getitem__(self, idx: int):
        return (
            self.user_indices[idx],
            self.item_indices[idx],
            self.weights[idx],
        )

class RecSysDataModule:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    def _resolve_split_dates(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return pd.Timestamp(self.args.train_end), pd.Timestamp(self.args.valid_end)

    def _build_feature_tables(
        self,
        users: pd.DataFrame,
        banners: pd.DataFrame,
    ) -> tuple[EncodedTable, EncodedTable]:
        user_table = encode_table(
            frame=users,
            id_column="user_id",
            categorical_columns=USER_CATEGORICAL_COLUMNS,
            numerical_columns=USER_NUMERIC_COLUMNS,
        )
        item_table = encode_table(
            frame=banners,
            id_column="banner_id",
            categorical_columns=ITEM_CATEGORICAL_COLUMNS,
            numerical_columns=ITEM_NUMERIC_COLUMNS,
        )
        return user_table, item_table

    def _build_train_loader(self, train_tensors: PairTensors) -> DataLoader:
        return DataLoader(
            PairDataset(train_tensors),
            batch_size=self.args.batch_size,
            shuffle=self.args.shuffle_train,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            drop_last=self.args.drop_last_train,
        )
    
    def _pairs_to_tensors(self, pairs: pd.DataFrame, user_table: EncodedTable, item_table: EncodedTable) -> PairTensors:
        mapped = pairs.copy()
        mapped["user_index"] = mapped["user_id"].map(user_table.id_to_row)
        mapped["item_index"] = mapped["banner_id"].map(item_table.id_to_row)
        mapped = mapped.dropna(subset=["user_index", "item_index"]).reset_index(drop=True)
        if mapped.empty:
            raise ValueError("No training pairs remained after mapping ids to feature tables.")
        return PairTensors(
            user_indices=torch.tensor(mapped["user_index"].astype(np.int64).to_numpy(), dtype=torch.long),
            item_indices=torch.tensor(mapped["item_index"].astype(np.int64).to_numpy(), dtype=torch.long),
            weights=torch.tensor(mapped["weight"].astype(np.float32).to_numpy(), dtype=torch.float32),
        )
    
    def _build_positive_pairs(self, interactions: pd.DataFrame) -> pd.DataFrame:
        positives = interactions.loc[
            interactions["clicks"].gt(0),
            ["user_id", "banner_id", "clicks"],
        ]

        if positives.empty:
            raise ValueError("No positive interactions with clicks > 0 were found.")

        pairs = positives.groupby(
            ["user_id", "banner_id"],
            as_index=False,
            sort=True,
        )["clicks"].sum()
        pairs["weight"] = np.log1p(pairs["clicks"].to_numpy()).astype(np.float32)

        return pairs


    def _load_tables(self, data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        users = pd.read_csv(data_dir / "users.csv")
        banners = pd.read_csv(data_dir / "banners.csv", parse_dates=["created_at"])
        interactions = pd.read_csv(
            data_dir / "banner_interactions.csv",
            parse_dates=["event_date"],
        )
        return users, banners, interactions


    def _prepare_banners(self, banners: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
        prepared = banners.copy()
        prepared["banner_age_days"] = (
            (reference_date - prepared["created_at"]).dt.days.clip(lower=0).astype(np.float32)
        )
        return prepared


    def _split_interactions(
        self,
        interactions: pd.DataFrame,
        train_end: pd.Timestamp,
        valid_end: pd.Timestamp,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_df = interactions[interactions["event_date"] <= train_end].copy()
        valid_df = interactions[
            (interactions["event_date"] > train_end)
            & (interactions["event_date"] <= valid_end)
        ].copy()
        test_df = interactions[interactions["event_date"] > valid_end].copy()
        return train_df, valid_df, test_df




    def setup(self) -> DataBundle:
        train_end, valid_end = self._resolve_split_dates()

        users, banners, interactions = self._load_tables(self.args.data_dir)
        banners = self._prepare_banners(banners, reference_date=train_end)

        train_df, valid_df, test_df = self._split_interactions(interactions, train_end, valid_end)
        train_pairs = self._build_positive_pairs(train_df)
        valid_pairs = self._build_positive_pairs(valid_df)
        test_pairs = self._build_positive_pairs(test_df)

        user_table, item_table = self._build_feature_tables(users, banners)
        train_tensors = self._pairs_to_tensors(train_pairs, user_table, item_table)
        train_loader = self._build_train_loader(train_tensors)

        return DataBundle(
            user_table=user_table,
            item_table=item_table,
            train_loader=train_loader,
            train_pairs=train_pairs,
            valid_pairs=valid_pairs,
            test_pairs=test_pairs,
        )
