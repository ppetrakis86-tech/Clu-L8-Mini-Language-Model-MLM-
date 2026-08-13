import torch
import torch.nn as nn
import math


# --- 1. Token Embedding ---
# for every token id make a vector of 768 dementions
# Πίνακας 93x512 που μαθαίνει κατά το training

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

    def forward(self, x):
        # x: [batch_size, seq_len] → ids
        # output: [batch_size, seq_len, hidden_dim] → vectors
        return self.embedding(x)


# --- 2. Positional Encoding ---
# add a  vector for every possition (0, 1, 2, ...) for the attention and model to know what is following
# we use sin/cos from "Attention is All You Need".

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=512):
        super().__init__()

        # position table [max_len, hidden_dim]
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # odd dimentions
        pe[:, 1::2] = torch.cos(position * div_term)  # even dimentions

        # save as buffer when it not learn.
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, hidden_dim]

    def forward(self, x):
        # x: [batch_size, seq_len, hidden_dim]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


# --- 3. Embedding combination ---
# Token Embedding + Positional Encoding together

class Embedding(nn.Module):
    def __init__(self, vocab_size, hidden_dim, max_len=512):
        super().__init__()
        self.token = TokenEmbedding(vocab_size, hidden_dim)
        self.position = PositionalEncoding(hidden_dim, max_len)

    def forward(self, x):
        # 1. Token ids → vectors
        x = self.token(x)
        # 2. Προσθήκη positional info
        x = self.position(x)
        return x
