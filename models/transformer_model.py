import math
from functools import cached_property

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = 1 / math.sqrt(self.head_dim)

    def forward(self, x, mask=None):
        batch_size, seq_length, embed_dim = x.size()

        qkv = self.qkv_proj(x)  # (batch_size, seq_length, embed_dim * 3)
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, 3 * head_dim)

        q, k, v = qkv.chunk(3, dim=-1)  # each will be (batch_size, num_heads, seq_length, head_dim)

        attn_scores = torch.einsum('bnqd,bnkd->bnqk', q,
                                   k) * self.scale  # (batch_size, num_heads, seq_length, seq_length)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_probs = nn.functional.softmax(attn_scores, dim=-1)  # (batch_size, num_heads, seq_length, seq_length)

        attn_output = torch.einsum('bnqk,bnvd->bnqd', attn_probs, v)  # (batch_size, num_heads, seq_length, head_dim)
        attn_output = attn_output.reshape(batch_size, seq_length, embed_dim)  # (batch_size, seq_length, embed_dim)

        output = self.o_proj(attn_output)  # (batch_size, seq_length, embed_dim)
        return output


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.attention(x, mask)
        x = self.layernorm1(x + self.dropout(attn_output))
        ff_output = self.feedforward(x)
        x = self.layernorm2(x + self.dropout(ff_output))
        return x


class CustomTransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, hidden_dim, output_dim, dropout=0.1,
                 finetune=True):
        super(CustomTransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)])
        self.fc_out = nn.Linear(embed_dim, output_dim)
        self.finetune = finetune

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).T
        mask = mask.int()
        return mask

    def forward(self, x):
        batch_size, seq_len = x.size()
        mask = self.generate_square_subsequent_mask(seq_len).to(x.device)
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)
        if self.finetune:
            x = x.mean(dim=1)
            x = self.fc_out(x)
            return x.squeeze(1)
        else:
            x = self.fc_out(x)
            return x.squeeze(2)

    @cached_property
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
