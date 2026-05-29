import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.config import Config
from src.dataset import load_and_split, normalize
from src.models import SLTTCN


def predict_at(time_index: int):
    """
    Given a time index in the TEST set, show actual vs predicted flow.
    time_index: 0 means first sample in test set
    """
    cfg = Config()

    # --- Load and preprocess data ---
    train_raw, val_raw, test_raw = load_and_split(cfg)
    _, _, test_n, mu, sigma = normalize(train_raw, val_raw, test_raw)

    # --- Load trained model ---
    device = torch.device("cpu")
    model = SLTTCN(cfg)
    model.load_state_dict(torch.load(
        f"checkpoints/slttcn_{cfg.dataset}_best.pt",
        map_location=device,
        weights_only=True
    ))
    model.eval()

    # --- Grab one window at time_index ---
    x = test_n[time_index : time_index + cfg.input_len]        # (12, 307)
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # (1, 12, 307)

    # --- Get prediction ---
    with torch.no_grad():
        pred_norm = model(x_tensor).squeeze(0).numpy()        # (3, 307)

    # --- Invert normalization ---
    pred   = pred_norm * sigma + mu                            # (3, 307)

    # --- Get actual targets ---
    actual = []
    for h in cfg.horizons:
        target_idx = time_index + cfg.input_len + h - 1
        if target_idx < len(test_n):
            actual.append(test_n[target_idx] * sigma + mu)    # (307,)
        else:
            actual.append(np.full(307, np.nan))
    actual = np.array(actual)                                  # (3, 307)

    # --- Print summary ---
    print(f"\nPrediction at test index: {time_index}")
    print(f"{'Horizon':<12} {'Pred (mean)':>14} {'Actual (mean)':>14} {'MAE':>10}")
    print("-" * 55)
    for i, h in enumerate(cfg.horizons):
        p = pred[i].mean()
        a = actual[i].mean()
        mae = np.abs(pred[i] - actual[i]).mean()
        print(f"{h} steps ({h*5:>3}min) {p:>14.2f} {a:>14.2f} {mae:>10.2f}")

    # --- Plot: predicted vs actual for a few sensors ---
    sensors_to_plot = [0, 50, 150, 250]
    horizon_labels  = [f"H{h} ({h*5}min)" for h in cfg.horizons]

    fig, axes = plt.subplots(len(sensors_to_plot), 1,
                             figsize=(10, 3 * len(sensors_to_plot)))
    fig.suptitle(f"Predicted vs Actual — Test Index {time_index}", fontweight="bold")

    for ax, sid in zip(axes, sensors_to_plot):
        ax.plot(cfg.horizons, pred[:, sid],   "o--", color="tomato",    label="Predicted")
        ax.plot(cfg.horizons, actual[:, sid], "o-",  color="steelblue", label="Actual")
        ax.set_title(f"Sensor {sid}")
        ax.set_xlabel("Horizon (steps)")
        ax.set_ylabel("Flow")
        ax.set_xticks(cfg.horizons)
        ax.set_xticklabels(horizon_labels)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"results/prediction_t{time_index}.png", dpi=150)
    plt.show()
    print(f"\nPlot saved to results/prediction_t{time_index}.png")


if __name__ == "__main__":
    import sys
    # Pass time index as argument, default to 0
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    predict_at(t)