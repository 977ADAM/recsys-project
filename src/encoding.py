import torch
from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class EncodedTable:
    ids: list[str]
    id_to_row: dict[str, int]
    categorical: torch.Tensor
    numerical: torch.Tensor
    cardinalities: list[int]

def encode_table(
    frame: pd.DataFrame,
    id_column: str,
    categorical_columns: list[str],
    numerical_columns: list[str],
) -> EncodedTable:
    table = frame.drop_duplicates(subset=id_column).sort_values(id_column).reset_index(drop=True).copy()
    for column in categorical_columns:
        table[column] = table[column].fillna("__missing__").astype(str)

    numeric_frame = table[numerical_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    numeric_mean = numeric_frame.mean()
    numeric_std = numeric_frame.std(ddof=0).replace(0.0, 1.0)
    standardized_numeric = ((numeric_frame - numeric_mean) / numeric_std).astype(np.float32)

    categorical_arrays: list[np.ndarray] = []
    cardinalities: list[int] = []
    for column in categorical_columns:
        values = table[column].astype(str)
        vocabulary = {value: idx + 1 for idx, value in enumerate(sorted(values.unique()))}
        encoded = values.map(vocabulary).fillna(0).astype(np.int64).to_numpy()
        categorical_arrays.append(encoded)
        cardinalities.append(len(vocabulary) + 1)

    categorical_matrix = np.stack(categorical_arrays, axis=1).astype(np.int64)
    ids = table[id_column].astype(str).tolist()

    return EncodedTable(
        ids=ids,
        id_to_row={entity_id: idx for idx, entity_id in enumerate(ids)},
        categorical=torch.tensor(categorical_matrix, dtype=torch.long),
        numerical=torch.tensor(standardized_numeric.to_numpy(), dtype=torch.float32),
        cardinalities=cardinalities,
    )