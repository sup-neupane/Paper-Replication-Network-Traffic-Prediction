import os
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np


class EarlyStopping:
    """Stop training when val loss stops improving for `patience` epochs."""

    def __init__(self, patience: int, checkpoint_path: str):
        self.patience = patience
        self.path = checkpoint_path
        self.best_loss = float("inf")
        self.counter = 0
        self.stopped = False

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)   # save best weights
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True
        return self.stopped


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        # x: (batch, input_len, N)
        # y: (batch, n_horizons, N)

        optimizer.zero_grad()
        pred = model(x)                     # (batch, n_horizons, N)
        loss = criterion(pred, y)           # MSE — L2 reg handled by weight_decay in Adam
        loss.backward()

        # Gradient clipping prevents exploding gradients in Transformer
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        total_loss += criterion(pred, y).item()

    return total_loss / len(loader)


def run_training(model, train_loader, val_loader, cfg):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(cfg.checkpoint_dir,
                                   f"slttcn_{cfg.dataset}_best.pt")

    # Adam with weight_decay implements L2 regularization (Eq. 16-17)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )

    # Learning rate scheduler: halve lr if val loss stagnates for 5 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    criterion = nn.MSELoss()
    stopper   = EarlyStopping(cfg.patience, checkpoint_path)

    print(f"Training on {device} | Dataset: {cfg.dataset.upper()}")
    print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12} | {'LR':>10}")
    print("-" * 50)

    train_losses, val_losses = [], []

    for epoch in range(1, cfg.max_epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer,
                                  criterion, device, cfg.grad_clip)
        va_loss = validate(model, val_loader, criterion, device)

        scheduler.step(va_loss)
        train_losses.append(tr_loss)
        val_losses.append(va_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"{epoch:>6} | {tr_loss:>12.4f} | {va_loss:>12.4f} | {lr_now:>10.6f}")

        if stopper.step(va_loss, model):
            print(f"\nEarly stopping at epoch {epoch}. Best val loss: {stopper.best_loss:.4f}")
            break

    # Reload best weights before returning
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    return model, train_losses, val_losses