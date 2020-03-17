import torch
import torch.nn as nn

from torch_code.BiGraphConv import BipartiteGraphConvolution
from torch_code.embedding_modules import EmbeddingModule, EdgeEmbeddingModule
from torch_code.output_module import OutputModule

from torch_code.pre_norm_layer import PreNormLayer, PreNormException


class NeuralNet(nn.Module):
    def __init__(self, device):
        super(NeuralNet, self).__init__()
        self.emb_size = 64
        self.cons_nfeats = 5
        self.edge_nfeats = 1
        self.var_nfeats = 19


        self.cons_embedding = EmbeddingModule(self.cons_nfeats, self.emb_size, device).to(device)
        self.edge_embedding = EdgeEmbeddingModule(self.edge_nfeats, device).to(device)
        self.var_embedding = EmbeddingModule(self.var_nfeats, self.emb_size, device).to(device)

        self.conv_v_to_c = BipartiteGraphConvolution(self.emb_size, right_to_left=True, device=device)
        self.conv_c_to_v = BipartiteGraphConvolution(self.emb_size, device=device)

        self.output_module = OutputModule(self.emb_size)

    def forward(self, inputs):
        constraint_features, edge_indices, edge_features, variable_features, n_cons_per_sample, n_vars_per_sample = inputs
        n_cons_total = torch.sum(n_cons_per_sample)
        n_vars_total = torch.sum(n_vars_per_sample)

        # EMBEDDINGS
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        # Convolutions
        constraint_features = self.conv_v_to_c((
            constraint_features, edge_indices, edge_features, variable_features, n_cons_total))
        constraint_features = nn.functional.relu(constraint_features)

        variable_features = self.conv_c_to_v((
            constraint_features, edge_indices, edge_features, variable_features, n_vars_total))
        variable_features = nn.functional.relu(variable_features)

        del constraint_features

        output = self.output_module(variable_features)

        output = output.reshape([1, -1])

        return output

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

