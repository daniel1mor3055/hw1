import torch
from torch import nn


class CustomLSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(CustomLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.hidden_dim = hidden_dim
        self.input_dim = embed_dim

        # LSTM weights and biases
        self.Wf = nn.Parameter(torch.empty(hidden_dim, embed_dim + hidden_dim))
        self.bf = nn.Parameter(torch.zeros(hidden_dim, 1))
        self.Wi = nn.Parameter(torch.empty(hidden_dim, embed_dim + hidden_dim))
        self.bi = nn.Parameter(torch.zeros(hidden_dim, 1))
        self.Wo = nn.Parameter(torch.empty(hidden_dim, embed_dim + hidden_dim))
        self.bo = nn.Parameter(torch.zeros(hidden_dim, 1))
        self.Wc = nn.Parameter(torch.empty(hidden_dim, embed_dim + hidden_dim))
        self.bc = nn.Parameter(torch.zeros(hidden_dim, 1))

        # Output layer weights and biases
        self.Wy = nn.Parameter(torch.empty(output_dim, hidden_dim))
        self.by = nn.Parameter(torch.zeros(output_dim, 1))

        # Initialize weights using Xavier initialization
        nn.init.xavier_uniform_(self.Wf)
        nn.init.xavier_uniform_(self.Wi)
        nn.init.xavier_uniform_(self.Wo)
        nn.init.xavier_uniform_(self.Wc)
        nn.init.xavier_uniform_(self.Wy)

    def sigmoid(self, x):
        return torch.sigmoid(x)

    def tanh(self, x):
        return torch.tanh(x)

    def forward(self, texts):
        seq_len, batch_size = texts.size()
        embedded = self.embedding(texts.T).permute(
            1, 2, 0
        )  # (seq_len,embed_dim, batch_size)

        h = torch.zeros(self.hidden_dim, batch_size).to(texts.device)
        c = torch.zeros(self.hidden_dim, batch_size).to(texts.device)

        for t in range(seq_len):
            x = embedded[t, :, :]  # (embed_dim, batch_size)
            concat = torch.cat((h, x), dim=0)  # (hidden_dim + embed_dim, batch_size)

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

        h = h.permute(1, 0)  # (batch_size, hidden_dim)
        y = torch.matmul(h, self.Wy.t()) + self.by.t()  # (batch_size, output_dim)
        return y.squeeze(1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
