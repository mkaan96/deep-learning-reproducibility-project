import torch.nn as nn

from torch_code.pre_norm_layer import PreNormLayer


class EmbeddingModule(nn.Module):
    """
    This class will be used for both variable and constraint embedding
    """
    def __init__(self, n_feats, emb_size):
        super().__init__()
        self.pre_norm_layer = PreNormLayer(n_units=n_feats).cuda()
        self.nn_layer_1 = nn.Linear(n_feats, emb_size).cuda()
        nn.init.orthogonal_(self.nn_layer_1.weight)

        self.nn_layer_2 = nn.Linear(emb_size, emb_size).cuda()
        nn.init.orthogonal_(self.nn_layer_2.weight)

    def forward(self, input):
        pre_norm_output = self.pre_norm_layer(input)
        layer_1_output = self.nn_layer_1(pre_norm_output)
        layer_1_output = nn.functional.relu(layer_1_output)

        layer_2_output = self.nn_layer_2(layer_1_output)
        layer_2_output = nn.functional.relu(layer_2_output)
        del pre_norm_output
        del layer_1_output
        return layer_2_output


class EdgeEmbeddingModule(nn.Module):
    """
    This class will only be used for edge embedding
    """
    def __init__(self, n_feats):
        super().__init__()
        self.pre_norm_layer = PreNormLayer(n_units=n_feats)

    def forward(self, input):
        return self.pre_norm_layer(input)