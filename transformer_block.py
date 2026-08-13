import torch
import torch.nn as nn
from attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()

        # --- Attention ---
        self.attention = MultiHeadAttention(hidden_dim, num_heads)

        # --- Feed Forward Network ---
        # 2 linear layers with GELU activation
        # hidden_dim -> ffn_dim -> hidden_dim
        # (768 -> 3072 -> 768)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim)
        )

        # --- Layer Normalization ---
        # after each step normalize numbers
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # --- Dropout ---
        # random neurons deactivation at training to avoid overfitting
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 1. Attention + Residual Connection
        x = x + self.dropout(self.attention(self.norm1(x)))

        # 2. FFN + Residual Connection
        x = x + self.dropout(self.ffn(self.norm2(x)))

        return x
