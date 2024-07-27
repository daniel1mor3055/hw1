from functools import cached_property

import torch
from torch import nn


class CustomLSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, finetune=True):
        super(CustomLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.hidden_dim = hidden_dim
        self.input_dim = embed_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.finetune = finetune

        # Create layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            input_dim = embed_dim if i == 0 else hidden_dim
            self.layers.append(LSTMLayer(input_dim, hidden_dim))

        # Output layer weights and biases
        self.Wy = nn.Parameter(torch.empty(vocab_size, hidden_dim))
        self.by = nn.Parameter(torch.zeros(vocab_size, 1))

        # Initialize weights using Xavier initialization
        nn.init.xavier_uniform_(self.Wy)

    def forward(self, texts):
        batch_size, seq_len = texts.size()
        embedded = self.embedding(texts).permute(1, 2, 0)  # (seq_len, embed_dim, batch_size)

        h = [
            torch.zeros(self.hidden_dim, batch_size).to(texts.device)
            for _ in range(self.num_layers)
        ]
        c = [
            torch.zeros(self.hidden_dim, batch_size).to(texts.device)
            for _ in range(self.num_layers)
        ]

        if not self.finetune:
            y = torch.zeros(seq_len, batch_size, self.vocab_size).to(texts.device)  # (seq_len, batch_size, vocab_size)
            for t in range(seq_len):
                x = embedded[t, :, :]  # (embed_dim, batch_size)
                for i, layer in enumerate(self.layers):
                    h[i], c[i] = layer(x, h[i], c[i])
                    x = h[i]
                h_last = h[-1].permute(1, 0)  # (batch_size, hidden_dim)
                y[t] = torch.matmul(h_last, self.Wy.t()) + self.by.t()  # (batch_size, vocab_size)

            return y.permute(1, 0, 2)
        else:
            for t in range(seq_len):
                x = embedded[t, :, :]  # (embed_dim, batch_size)
                for i, layer in enumerate(self.layers):
                    h[i], c[i] = layer(x, h[i], c[i])
                    x = h[i]

            h_last = h[-1].permute(1, 0)  # (batch_size, hidden_dim)
            y = torch.matmul(h_last, self.Wy.t()) + self.by.t()  # (batch_size, output_dim)
            return y.squeeze(1)

    @cached_property
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LSTMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMLayer, self).__init__()
        self.hidden_dim = hidden_dim

        # LSTM weights and biases
        self.Wf = nn.Parameter(torch.empty(hidden_dim, input_dim + hidden_dim))
        self.bf = nn.Parameter(torch.zeros(hidden_dim, 1))
        self.Wi = nn.Parameter(torch.empty(hidden_dim, input_dim + hidden_dim))
        self.bi = nn.Parameter(torch.zeros(hidden_dim, 1))
        self.Wo = nn.Parameter(torch.empty(hidden_dim, input_dim + hidden_dim))
        self.bo = nn.Parameter(torch.zeros(hidden_dim, 1))
        self.Wc = nn.Parameter(torch.empty(hidden_dim, input_dim + hidden_dim))
        self.bc = nn.Parameter(torch.zeros(hidden_dim, 1))

        # Initialize weights using Xavier initialization
        nn.init.xavier_uniform_(self.Wf)
        nn.init.xavier_uniform_(self.Wi)
        nn.init.xavier_uniform_(self.Wo)
        nn.init.xavier_uniform_(self.Wc)

    def sigmoid(self, x):
        return torch.sigmoid(x)

    def tanh(self, x):
        return torch.tanh(x)

    def forward(self, x, h, c):
        concat = torch.cat((h, x), dim=0)  # (hidden_dim + input_dim, batch_size)

        ft = self.sigmoid(
            torch.matmul(self.Wf, concat) + self.bf
        )  # (hidden, batch_size)
        it = self.sigmoid(
            torch.matmul(self.Wi, concat) + self.bi
        )  # (hidden, batch_size)
        c_hat = self.tanh(
            torch.matmul(self.Wc, concat) + self.bc
        )  # (hidden, batch_size)
        c = ft * c + it * c_hat  # (hidden, batch_size)
        ot = self.sigmoid(
            torch.matmul(self.Wo, concat) + self.bo
        )  # (hidden, batch_size)
        h = ot * self.tanh(c)

        return h, c
