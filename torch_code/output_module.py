import torch.nn as nn


class OutputModule(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.emb_size = emb_size

        self.layer_1 = nn.Linear(self.emb_size, self.emb_size)
        nn.init.orthogonal_(self.layer_1.weight)

        self.layer_2 = nn.Linear(self.emb_size, 1)
        nn.init.orthogonal_(self.layer_2.weight)

    def forward(self, variable_features):
        output = self.layer_1(variable_features)
        output = nn.functional.relu(output)
        output = self.layer_2(output)
        output = nn.functional.relu(output)
        return output
