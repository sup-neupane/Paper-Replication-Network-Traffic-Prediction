import torch
import numpy as np
from src.config import Config


def inverse_normalize(y_norm: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Undo Z-score normalization to get back real traffic flow values."""
    return y_norm * sigma + mu


@torch.no_grad()
def get_predictions(model, loader, device):
    """Run model on entire loader. Returns preds and targets as numpy arrays."""
    model.eval()
    all_preds, all_targets = [], []

    for x, y in loader:
        x = x.to(device)
        pred = model(x).cpu().numpy()   # (batch, n_horizons, N)
        all_preds.append(pred)
        all_targets.append(y.numpy())

    preds   = np.concatenate(all_preds,   axis=0)   # (total_samples, n_horizons, N)
    targets = np.concatenate(all_targets, axis=0)

    return preds, targets


def compute_metrics(preds: np.ndarray, targets: np.ndarray):
    """
    Compute MAE and RMSE per horizon.
    preds, targets: (samples, n_horizons, N_sensors)
    Returns dict: {horizon_index: {"MAE": float, "RMSE": float}}
    """
    n_horizons = preds.shape[1]
    results = {}

    for h in range(n_horizons):
        p = preds[:, h, :].flatten()      # all samples, all sensors for horizon h
        t = targets[:, h, :].flatten()

        mae  = np.mean(np.abs(p - t))
        rmse = np.sqrt(np.mean((p - t) ** 2))
        results[h] = {"MAE": mae, "RMSE": rmse}

    return results


def evaluate(model, test_loader, cfg: Config, mu: float, sigma: float):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    preds_norm, targets_norm = get_predictions(model, test_loader, device)

    # Invert normalization — evaluate in original traffic flow units
    preds   = inverse_normalize(preds_norm,   mu, sigma)
    targets = inverse_normalize(targets_norm, mu, sigma)

    metrics = compute_metrics(preds, targets)

    print(f"\nTest Results — {cfg.dataset.upper()}")
    print(f"{'Horizon':>10} | {'MAE':>10} | {'RMSE':>10}")
    print("-" * 36)
    for h_idx, h_val in enumerate(cfg.horizons):
        m = metrics[h_idx]
        print(f"{h_val:>10} | {m['MAE']:>10.2f} | {m['RMSE']:>10.2f}")

    return metrics