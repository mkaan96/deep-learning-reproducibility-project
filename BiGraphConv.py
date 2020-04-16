import torch
import torch.nn as nn


class BipartiteGraphConvolution(nn.Module):
    """
    Partial bipartite graph convolution (either left-to-right or right-to-left).
    """

    def __init__(self, emb_size, right_to_left=False, device=None):
        super().__init__()
        self.iterations = 0
        self.device = device
        self.emb_size = emb_size
        self.right_to_left = right_to_left

        self.feature_module_left = nn.Linear(self.emb_size, self.emb_size)
        nn.init.orthogonal_(self.feature_module_left.weight)

        self.feature_module_edge = nn.Linear(1, self.emb_size, bias=False)
        nn.init.orthogonal_(self.feature_module_edge.weight)

        self.feature_module_right = nn.Linear(self.emb_size, self.emb_size, bias=False)
        nn.init.orthogonal_(self.feature_module_right.weight)

        self.feature_model_final = nn.Sequential(
            nn.BatchNorm1d(self.emb_size),
            nn.ReLU(),
            nn.Linear(self.emb_size, self.emb_size)
        )

        self.post_conv_module = nn.BatchNorm1d(self.emb_size)

        self.output_module = nn.Sequential(
            nn.Linear(2 * self.emb_size, self.emb_size),
            nn.ReLU(),
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU()
        )

    def forward(self, inputs):
        """
        Performs a partial graph convolution on the given bipartite graph.

        Inputs
        ------
        left_features: 2D float tensor
            Features of the left-hand-side nodes in the bipartite graph
        edge_indices: 2D int tensor
            Edge indices in left-right order
        edge_features: 2D float tensor
            Features of the edges
        right_features: 2D float tensor
            Features of the right-hand-side nodes in the bipartite graph
        scatter_out_size: 1D int tensor
            Output size (left_features.shape[0] or right_features.shape[0], unknown at compile time)

        """
        self.iterations += 1
        left_features, edge_indices, edge_features, right_features, scatter_out_size = inputs

        if self.right_to_left:
            scatter_dim = 0
            prev_features = left_features
        else:
            scatter_dim = 1
            prev_features = right_features

        joint_features = self.feature_module_edge(edge_features)
        joint_features.add_(self.feature_module_right(right_features)[edge_indices[1]])
        joint_features.add_(self.feature_module_left(left_features)[edge_indices[0]])
        joint_features = self.feature_model_final(joint_features)

        conv_output = torch.zeros([scatter_out_size, self.emb_size])\
            .to(self.device).index_add(0, edge_indices[scatter_dim], joint_features)

        conv_output = self.post_conv_module(conv_output)

        output = torch.cat((conv_output, prev_features), dim=1)

        return self.output_module(output)

    

