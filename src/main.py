import torch
import numpy as np
import random
import os

from src.config import Config
from src.dataset import build_dataloaders
from src.models import SLTTCN 
from src.train import run_training
from src.evaluate import evaluate


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    cfg = Config()
    set_seed(cfg.seed)
    os.makedirs(cfg.results_dir, exist_ok=True)

    # --- Data ---
    train_loader, val_loader, test_loader, mu, sigma = build_dataloaders(cfg)

    # --- Model ---
    model = SLTTCN(cfg)

    # --- Train ---
    model, train_losses, val_losses = run_training(
        model, train_loader, val_loader, cfg
    )

    # --- Evaluate ---
    metrics = evaluate(model, test_loader, cfg, mu, sigma)

    # --- Save results ---
    import json
    out = {
        "dataset": cfg.dataset,
        "horizons": cfg.horizons,
        "metrics": {str(k): v for k, v in metrics.items()},
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    with open(f"{cfg.results_dir}/results_{cfg.dataset}.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nResults saved to {cfg.results_dir}/results_{cfg.dataset}.json")


if __name__ == "__main__":
    main()