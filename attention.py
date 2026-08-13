import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim πρέπει να διαιρείται με num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads  # 768 / 12 = 64

        # Q, K, V linear layers
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)

        # Output linear layer for combine 12 attention heads
        self.W_o = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # 1. calculate Q, K, V atention
        Q = self.W_q(x)  # [batch, seq_len, 768]
        K = self.W_k(x)
        V = self.W_v(x)

        # 2. 12 heads seperation
        # [batch, seq_len, 768] → [batch, 12, seq_len, 64]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Scaled Dot-Product Attention
        # scores for attention the tokens betwin them
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 4. Causal Mask — every token look the previous tokens 
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))

        # 5. Softmax
        attn_weights = torch.softmax(scores, dim=-1)

        # 6. multiply with V
        attn_output = torch.matmul(attn_weights, V)  # [batch, 12, seq_len, 64]

        # 7. recombine heads back to 768
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)

        # 8. Output linear layer
        return self.W_o(attn_output)
