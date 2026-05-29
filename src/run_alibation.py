from src.config import Config
from src.dataset import build_dataloaders
from src.models import SLTTCN, MFE, TCF, InputProjection, SLTTCN
from src.train import run_training
from src.evaluate import evaluate
import torch
import torch.nn as nn


class OnlyTCF(nn.Module):
    """Full model with MFE removed. Input → TCF → Output."""
    def __init__(self, cfg):
        super().__init__()
        n_sensors = cfg.num_sensors[cfg.dataset]
        self.input_proj = InputProjection(n_sensors, cfg.d_model)
        # Skip MFE entirely
        self.tcf = TCF(cfg.d_model, cfg.tcn_channels,
                       cfg.tcn_kernel_size, cfg.tcn_layers, cfg.dropout)
        self.output_heads = nn.ModuleList([
            nn.Linear(cfg.tcn_channels, n_sensors)
            for _ in range(len(cfg.horizons))
        ])

    def forward(self, x):
        x = self.input_proj(x)
        x = self.tcf(x)
        last = x[:, -1, :]
        return torch.stack([h(last) for h in self.output_heads], dim=1)


class OnlyMFE(nn.Module):
    """Full model with TCF removed. Input → MFE → Output."""
    def __init__(self, cfg):
        super().__init__()
        n_sensors = cfg.num_sensors[cfg.dataset]
        self.input_proj = InputProjection(n_sensors, cfg.d_model)
        self.mfe = MFE(cfg.d_model, cfg.n_heads, cfg.d_ff,
                       cfg.dropout, cfg.n_transformer_layers)
        # Skip TCF; output directly from d_model
        self.output_heads = nn.ModuleList([
            nn.Linear(cfg.d_model, n_sensors)
            for _ in range(len(cfg.horizons))
        ])

    def forward(self, x):
        x = self.input_proj(x)
        x = self.mfe(x)
        last = x[:, -1, :]
        return torch.stack([h(last) for h in self.output_heads], dim=1)


if __name__ == "__main__":
    cfg = Config()
    train_loader, val_loader, test_loader, mu, sigma = build_dataloaders(cfg)

    experiments = {
        "MFE+TCF (Full)": SLTTCN(cfg),
        "Only TCF":        OnlyTCF(cfg),
        "Only MFE":        OnlyMFE(cfg),
    }

    for name, model in experiments.items():
        print(f"\n{'='*50}")
        print(f"Running: {name}")
        cfg.checkpoint_dir = f"checkpoints/{name.replace(' ', '_')}"
        model, _, _ = run_training(model, train_loader, val_loader, cfg)
        evaluate(model, test_loader, cfg, mu, sigma)