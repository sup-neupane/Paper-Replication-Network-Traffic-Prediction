import torch
import torch.nn as nn
import math


class MFE(nn.Module):
    """
    Multi-head Feature Extraction module.
    Implements one Transformer encoder block:
      MultiHeadSelfAttention → Add&Norm → FFN → Add&Norm

    Input shape:  (batch, seq_len, d_model)
    Output shape: (batch, seq_len, d_model)
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float, n_layers: int = 1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """
    Single Transformer encoder block implementing equations (1)-(8) from the paper.
    """

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn    = nn.MultiheadAttention(d_model, n_heads,
                                              dropout=dropout, batch_first=True)
        self.ffn     = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.drop1   = nn.Dropout(dropout)
        self.drop2   = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, d_model)

        # --- Self-attention sublayer (Eq. 1-4) ---
        attn_out, _ = self.attn(x, x, x)          # Q=K=V=x (self-attention)
        x = self.norm1(x + self.drop1(attn_out))  # residual + LayerNorm (Eq. 6-7)

        # --- Feedforward sublayer (Eq. 5) ---
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.drop2(ffn_out))   # residual + LayerNorm

        return x
    
# src/model.py  (Part 2 — TCF, append to same file)


class CausalDilatedConv(nn.Module):
    """
    One causal dilated convolutional layer with padding.

    Causal means: Y_t depends only on Z_{t}, Z_{t-1}, ..., Z_{t-d*(K-1)}
    We implement causality by padding only on the LEFT (past side).

    Implements Eq. (9) and (10) from the paper.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        # Left-padding size to ensure output length == input length
        self.pad = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0          # we handle padding manually for strict causality
        )

    def forward(self, x):
        # x: (batch, channels, seq_len)
        x = torch.nn.functional.pad(x, (self.pad, 0))   # pad left only
        return self.conv(x)


class TCFBlock(nn.Module):
    """
    One residual block in TCF: CausalDilatedConv → BN → ReLU → Dropout + residual.
    Implements Eq. (11) and (12).

    If in_channels != out_channels, a 1x1 conv is used for the residual path
    (standard practice when channel dimensions change).
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.conv   = CausalDilatedConv(in_channels, out_channels, kernel_size, dilation)
        self.bn     = nn.BatchNorm1d(out_channels)
        self.relu   = nn.ReLU()
        self.drop   = nn.Dropout(dropout)

        # 1x1 conv to match dimensions for residual path if needed
        self.residual_proj = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        # x: (batch, in_channels, seq_len)
        out = self.drop(self.relu(self.bn(self.conv(x))))
        return out + self.residual_proj(x)         # Eq. (11)


class TCF(nn.Module):
    """
    Full Temporal Convolution Fusion module.
    Stacks 4 TCFBlocks with dilation factors [1, 2, 4, 8].

    Input shape:  (batch, seq_len, d_model)   — from MFE output
    Output shape: (batch, seq_len, tcn_channels)
    """

    def __init__(self, in_channels: int, tcn_channels: int,
                 kernel_size: int, n_layers: int, dropout: float):
        super().__init__()

        # Build exponentially dilated stack: d = 1, 2, 4, 8, ...
        dilations = [2 ** i for i in range(n_layers)]    # [1, 2, 4, 8]

        layers = []
        for i, d in enumerate(dilations):
            ch_in  = in_channels if i == 0 else tcn_channels
            ch_out = tcn_channels
            layers.append(TCFBlock(ch_in, ch_out, kernel_size, d, dropout))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x from MFE: (batch, seq_len, d_model)
        x = x.permute(0, 2, 1)           # → (batch, d_model, seq_len) for Conv1d
        x = self.network(x)               # → (batch, tcn_channels, seq_len)
        x = x.permute(0, 2, 1)           # → (batch, seq_len, tcn_channels)
        return x
    

# src/model.py  (Part 3 — full model, append to same file)


class InputProjection(nn.Module):
    """
    The raw input shape is (batch, seq_len, N_sensors).
    We need to project each time step from N_sensors → d_model.
    This is a learned linear embedding.
    """

    def __init__(self, n_sensors: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(n_sensors, d_model)

    def forward(self, x):
        # x: (batch, seq_len, n_sensors)
        return self.proj(x)    # → (batch, seq_len, d_model)


class SLTTCN(nn.Module):
    """
    Full model: InputProjection → MFE → TCF → MultiHorizonOutputHead

    The output head maps the final time step's representation to predictions
    for each target horizon. Since we're doing multi-horizon forecasting with
    separate labels per horizon, we use one linear layer per horizon.
    """

    def __init__(self, cfg):
        super().__init__()
        n_sensors   = cfg.num_sensors[cfg.dataset]
        d_model     = cfg.d_model
        tcn_channels = cfg.tcn_channels
        n_horizons  = len(cfg.horizons)

        # 1. Input embedding: sensors → d_model
        self.input_proj = InputProjection(n_sensors, d_model)

        # 2. Transformer encoder (MFE)
        self.mfe = MFE(
            d_model=d_model,
            n_heads=cfg.n_heads,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
            n_layers=cfg.n_transformer_layers
        )

        # 3. Temporal convolution (TCF)
        self.tcf = TCF(
            in_channels=d_model,
            tcn_channels=tcn_channels,
            kernel_size=cfg.tcn_kernel_size,
            n_layers=cfg.tcn_layers,
            dropout=cfg.dropout
        )

        # 4. Output heads — one per horizon (Eq. 13)
        # Takes the last time step of TCF output and predicts all N sensors
        self.output_heads = nn.ModuleList([
            nn.Linear(tcn_channels, n_sensors)
            for _ in range(n_horizons)
        ])

    def forward(self, x):
        # x: (batch, input_len, n_sensors)

        x = self.input_proj(x)      # (batch, input_len, d_model)
        x = self.mfe(x)             # (batch, input_len, d_model)
        x = self.tcf(x)             # (batch, input_len, tcn_channels)

        # Use the LAST time step as the summary representation
        last = x[:, -1, :]          # (batch, tcn_channels)

        # One prediction per horizon
        preds = torch.stack(
            [head(last) for head in self.output_heads],
            dim=1
        )                           # (batch, n_horizons, n_sensors)

        return preds


        