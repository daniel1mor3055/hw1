import torch
from torch import nn
from s4 import S4  # Assuming the S4 model is available via a Python module or you've installed it using the repository

class CustomS4Model(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, num_layers):
        super(CustomS4Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Initialize S4 layers
        self.s4_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.s4_layers.append(S4(d_model=embed_dim, d_state=hidden_dim))

        self.fc_out = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.s4_layers:
            x = layer(x)
        x = x.mean(dim=1)
        x = self.fc_out(x)
        return x.squeeze(1)

    @property
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
