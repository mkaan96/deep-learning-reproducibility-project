import torch
import torch.nn as nn
from torch_code.embedding_modules import EmbeddingModule, EdgeEmbeddingModule

from torch_code.pre_norm_layer import PreNormLayer, PreNormException


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.emb_size = 64
        self.cons_nfeats = 5
        self.edge_nfeats = 1
        self.var_nfeats = 19

        self.cons_embedding = EmbeddingModule(self.cons_nfeats, self.emb_size).cuda()
        self.edge_embedding = EdgeEmbeddingModule(self.edge_nfeats).cuda()
        self.var_embedding = EmbeddingModule(self.var_nfeats, self.emb_size).cuda()

    def forward(self, inputs):
        constraint_features, edge_indices, edge_features, variable_features, n_cons_per_sample, n_vars_per_sample = inputs

        # EMBEDDINGS
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        # Constraint, edge and variable features need to be convoluted.

        return []

    def pre_train_init(self):
        self.pre_train_init_rec()

    def pre_train_init_rec(self):
        layers = [module for module in self.modules() if type(module) != nn.Sequential]
        for layer in layers:
            if isinstance(layer, PreNormLayer):
                layer.start_updates()

    def pre_train(self,  *args, **kwargs):
        try:
            self.forward(*args, **kwargs)
            return False
        except PreNormException:
            return True

    def pre_train_next(self):
        return self.pre_train_next_rec()

    def pre_train_next_rec(self):
        layers = [module for module in self.modules() if type(module) != nn.Sequential]
        for layer in layers:
            if isinstance(layer, PreNormLayer) and layer.waiting_updates and layer.received_updates:
                layer.stop_updates()
                return layer
        return None

