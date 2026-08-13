import torch
import torch.nn as nn
from embeddings import Embedding
from transformer_block import TransformerBlock


class MLM(nn.Module):
    def __init__(
        self,
        vocab_size=93,
        hidden_dim=768,
        num_heads=12,
        ffn_dim=3072,
        num_layers=12,
        max_len=512,
        dropout=0.1,
    ):
        super().__init__()

        # --- 1. Embedding (token + positional) ---
        self.embedding = Embedding(vocab_size, hidden_dim, max_len)

        # --- 2. stack of 12 Transformer Blocks ---
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        # --- 3. Final Layer Norm ---
        self.norm = nn.LayerNorm(hidden_dim)

        # --- 4. Output Head ---
        # hidden_dim to vocab_size (chances for each next token)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # x: [batch, seq_len] to token ids

        # 1. Embeddings
        x = self.embedding(x)  # [batch, seq_len, hidden_dim]

        # 2. check all 12 blocks
        for block in self.blocks:
            x = block(x)

        # 3. Final norm
        x = self.norm(x)

        # 4 next token prediction
        logits = self.head(x)  # [batch, seq_len, vocab_size]

        return logits

    def count_params(self):
        return sum(p.numel() for p in self.parameters())
