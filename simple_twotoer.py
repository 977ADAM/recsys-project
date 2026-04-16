import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from rich.console import Console
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


console = Console()
data_dir = Path("data/raw")
train_end = pd.Timestamp("2026-02-28")
valid_end = pd.Timestamp("2026-03-15")

# Load datasets

users = pd.read_csv(data_dir / "users.csv")
if not users.empty:
    console.print(f"Loaded {len(users)} users")
else:
    raise console.print("No users found in the dataset")

banners = pd.read_csv(data_dir / "banners.csv", parse_dates=["created_at"])
if not banners.empty:
    console.print(f"Loaded {len(banners)} banners")
else:
    raise console.print("No banners found in the dataset")

interactions = pd.read_csv(data_dir / "banner_interactions.csv", parse_dates=["event_date"])
if not interactions.empty:
    console.print(f"Loaded {len(interactions)} interactions")
else:
    raise console.print("No interactions found in the dataset")

# split data

train_df = interactions[interactions["event_date"] <= train_end].copy()
if not train_df.empty:
    console.print(f"Training set: {len(train_df)} interactions")

valid_df = interactions[(interactions["event_date"] > train_end) & (interactions["event_date"] <= valid_end)].copy()
if not valid_df.empty:
    console.print(f"Validation set: {len(valid_df)} interactions")

test_df = interactions[interactions["event_date"] > valid_end].copy()
if not test_df.empty:
    console.print(f"Test set: {len(test_df)} interactions")

# Build positive pairs

def build_positive_pairs(df: pd.DataFrame) -> pd.DataFrame:
    positives = df.loc[df["clicks"].gt(0), ["user_id", "banner_id", "clicks"]]

    if positives.empty:
        raise ValueError("No positive interactions with clicks > 0 were found.")
    
    pairs = positives.groupby(["user_id", "banner_id"], as_index=False, sort=True)["clicks"].sum()
    pairs["weight"] = np.log1p(pairs["clicks"].to_numpy()).astype(np.float32)

    return pairs

train_pairs = build_positive_pairs(train_df)
console.print(f"Built {len(train_pairs)} positive pairs from training data")

valid_pairs = build_positive_pairs(valid_df)
console.print(f"Built {len(valid_pairs)} positive pairs from validation data")

test_pairs = build_positive_pairs(test_df)
console.print(f"Built {len(test_pairs)} positive pairs from test data")





























