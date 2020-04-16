import torch
import torch.nn as nn

from BiGraphConv import BipartiteGraphConvolution
from embedding_modules import EmbeddingModule, EdgeEmbeddingModule
from output_module import OutputModule


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
        if constraint_features is None:
            return None
        edge_features = self.edge_embedding(edge_features)
        if edge_features is None:
            return None
        variable_features = self.var_embedding(variable_features)
        if variable_features is None:
            return None

        # Convolutions
        constraint_features = self.conv_v_to_c((
            constraint_features, edge_indices, edge_features, variable_features, n_cons_total))
        if constraint_features is None:
            return None

        variable_features = self.conv_c_to_v((
            constraint_features, edge_indices, edge_features, variable_features, n_vars_total))
        if variable_features is None:
            return None

        output = self.output_module(variable_features)

        output = output.reshape([1, -1])

        return output

    
    def pad_output(self, output, n_vars_per_sample, pad_value=-1e8):
        n_vars_max = torch.max(n_vars_per_sample)
        output = torch.split(output, tuple(n_vars_per_sample), 1)
        output2 = []
        for x in output:
            newx = torch.nn.functional.pad(x,(0, n_vars_max.item() - x.shape[1]),'constant', pad_value)
            output2.append(newx)

        output = torch.cat(output2, 0)

        return output
