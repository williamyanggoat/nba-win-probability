import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from model.net import FEATURE_COLS

class GameDataset(Dataset):
    def __init__(self, parquet_path: str):
        df = pd.read_parquet(parquet_path)
        self.X = torch.tensor(
            df[FEATURE_COLS].fillna(0).values, dtype=torch.float32
        )
        self.y = torch.tensor(df["home_win"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]