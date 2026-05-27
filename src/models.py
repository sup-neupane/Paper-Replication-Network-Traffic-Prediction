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