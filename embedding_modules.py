import torch.nn as nn


class EmbeddingModule(nn.Module):
    """
    This class will be used for both variable and constraint embedding
    """
    def __init__(self, n_feats, emb_size, device):
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm1d(n_feats),
            nn.Linear(n_feats, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size),
            nn.ReLU()
        )

    def forward(self, input):
        return self.layer(input)


class EdgeEmbeddingModule(nn.Module):
    """
    This class will only be used for edge embedding
    """
    def __init__(self, n_feats, device):
        super().__init__()
        self.pre_norm_layer = nn.BatchNorm1d(n_feats)

    def forward(self, input):
        return self.pre_norm_layer(input)
