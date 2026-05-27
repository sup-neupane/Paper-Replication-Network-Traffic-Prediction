# src/dataset.py

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.config import Config


def load_and_split(cfg: Config):
    """
    Load raw .npz, extract traffic flow channel, split chronologically.
    Returns train/val/test arrays of shape (T_split, N_sensors).
    """
    raw = np.load(cfg.data_path[cfg.dataset])
    data = raw["data"]                          # shape: (T, N, C)
    data = data[:, :, cfg.feature_channel]      # shape: (T, N) — traffic flow only

    T = data.shape[0]
    train_end = int(T * cfg.train_ratio)
    val_end   = int(T * (cfg.train_ratio + cfg.val_ratio))

    train = data[:train_end]
    val   = data[train_end:val_end]
    test  = data[val_end:]

    return train, val, test


def normalize(train, val, test):
    """
    Z-score normalization. Stats computed ONLY from training set.
    This prevents data leakage into val/test.
    """
    mu    = train.mean()
    sigma = train.std()

    train_n = (train - mu) / sigma
    val_n   = (val   - mu) / sigma
    test_n  = (test  - mu) / sigma

    return train_n, val_n, test_n, mu, sigma


def make_windows(data: np.ndarray, input_len: int, horizons: list):
    """
    Sliding window construction.

    For each valid starting index t:
      - Input  X: data[t : t + input_len]         shape (input_len, N)
      - Labels Y: data[t + input_len + h - 1]     for each horizon h
                  shape (len(horizons), N)

    This implements multi-horizon prediction with separate labels per horizon.
    This resolves the replication ambiguity: instead of autoregressive rollout,
    we use a multi-output head (one per horizon). This is cleaner and avoids
    error accumulation.
    """
    max_horizon = max(horizons)
    X, Y = [], []

    # We need enough room for input + largest horizon
    for t in range(len(data) - input_len - max_horizon + 1):
        x = data[t : t + input_len]               # (input_len, N)
        # For each horizon, grab the single target time step
        y = np.stack([
            data[t + input_len + h - 1]            # (N,) per horizon
            for h in horizons
        ], axis=0)                                 # (num_horizons, N)
        X.append(x)
        Y.append(y)

    X = np.array(X, dtype=np.float32)             # (samples, input_len, N)
    Y = np.array(Y, dtype=np.float32)             # (samples, num_horizons, N)
    return X, Y


class TrafficDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.from_numpy(X)   # (samples, input_len, N)
        self.Y = torch.from_numpy(Y)   # (samples, num_horizons, N)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def build_dataloaders(cfg: Config):
    """
    Full pipeline: load → split → normalize → window → DataLoader.
    Returns train/val/test loaders and normalization stats (mu, sigma)
    needed to invert normalization during evaluation.
    """
    train_raw, val_raw, test_raw = load_and_split(cfg)
    train_n, val_n, test_n, mu, sigma = normalize(train_raw, val_raw, test_raw)

    X_tr, Y_tr = make_windows(train_n, cfg.input_len, cfg.horizons)
    X_va, Y_va = make_windows(val_n,   cfg.input_len, cfg.horizons)
    X_te, Y_te = make_windows(test_n,  cfg.input_len, cfg.horizons)

    print(f"Train windows: {X_tr.shape}  |  Val: {X_va.shape}  |  Test: {X_te.shape}")

    train_loader = DataLoader(TrafficDataset(X_tr, Y_tr),
                              batch_size=cfg.batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(TrafficDataset(X_va, Y_va),
                              batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(TrafficDataset(X_te, Y_te),
                              batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader, mu, sigma