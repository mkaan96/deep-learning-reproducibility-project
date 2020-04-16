import torch.nn as nn


class OutputModule(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.emb_size = emb_size

        self.layer = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(),
            nn.Linear(self.emb_size, 1),
        )

    def forward(self, variable_features):
        return self.layer(variable_features)
