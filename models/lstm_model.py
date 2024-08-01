from functools import cached_property

import torch
from torch import nn


class CustomLSTMModel(nn.Module):
    def __init__(
        self, vocab_size, embed_dim, hidden_dim, num_layers, output_dim, finetune=True
    ):
        super(CustomLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.hidden_dim = hidden_dim
        self.input_dim = embed_dim
        self.num_layers = num_layers
        self.finetune = finetune
        self.output_dim = output_dim

        # Built-in LSTM layer
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)

        # Output layer weights and biases
        self.output_layer = nn.Linear(
            in_features=hidden_dim, out_features=output_dim, bias=False
        )

    def forward(self, texts):
        batch_size, seq_len = texts.size()
        embedded = self.embedding(texts)  # (batch_size, seq_len, embed_dim)

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(texts.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(texts.device)

        lstm_out, (hn, cn) = self.lstm(
            embedded, (h0, c0)
        )  # (batch_size, seq_len, hidden_dim)

        if not self.finetune:
            y = self.output_layer(lstm_out)  # (batch_size, seq_len, output_dim)
            return y.permute(0, 2, 1)
        else:
            hn_last = hn[-1]  # (batch_size, hidden_dim)
            y = self.output_layer(hn_last)  # (batch_size, output_dim)
            return y

    @cached_property
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
