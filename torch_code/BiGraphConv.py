# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 11:28:53 2020

@author: Manisha
"""
import torch
import torch.nn as nn

# Fully connected neural network with one hidden layer
from torch_code.pre_norm_layer import PreNormLayer, PreNormException


class BipartiteGraphConvolution(nn.Module):
    """
    Partial bipartite graph convolution (either left-to-right or right-to-left).
    Notes on Keras Dense layer to pytorch :
    *Dense layer in Keras implements output = activation(dot(input, kernel) + bias)
    *Linear layer + point-wise non-linearity/activation = Dense Layer in Keras
    *Kernel initialier = statistical ddistribution to intilize weights

    """

    def __init__(self, emb_size, right_to_left=False):
        super().__init__()
        
        self.emb_size = emb_size
        self.right_to_left = right_to_left

        self.feature_module_left = nn.Linear(self.emb_size, self.emb_size)
        nn.init.orthogonal_(self.feature_module_left.weight)

        self.feature_module_edge = nn.Linear(1, self.emb_size, bias=False)
        nn.init.orthogonal_(self.feature_module_edge.weight)

        self.feature_module_right = nn.Linear(self.emb_size, self.emb_size, bias=False)
        nn.init.orthogonal_(self.feature_module_right.weight)

        self.feature_model_final_pre_norm = PreNormLayer(1, shift=False)
        self.feature_model_final_linear = nn.Linear(self.emb_size, self.emb_size)
        nn.init.orthogonal_(self.feature_model_final_linear.weight)

        self.post_conv_module = PreNormLayer(1, shift=False)

        self.output_module_layer_1 = nn.Linear(2*self.emb_size, self.emb_size)
        nn.init.orthogonal_(self.output_module_layer_1.weight)

        self.output_module_layer_2 = nn.Linear(self.emb_size, self.emb_size)
        nn.init.orthogonal_(self.output_module_layer_2.weight)

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

        Other parameters
        ----------------
        training: boolean
            Training mode indicator
        """
        left_features, edge_indices, edge_features, right_features, scatter_out_size = inputs

        if self.right_to_left:
            scatter_dim = 0
            prev_features = left_features
            is_left = True
        else:
            scatter_dim = 1
            prev_features = right_features
            is_left = False

        left_features = self.feature_module_left(left_features)[edge_indices[0]]
        edge_features = self.feature_module_edge(edge_features)
        right_features = self.feature_module_right(right_features)[edge_indices[1]]

        joint_features = left_features.item() + edge_features.item() + right_features.item()
        if is_left:
            del right_features
        else:
            del left_features
        del edge_features
        joint_features = self.feature_model_final_pre_norm(joint_features)
        joint_features = nn.functional.relu(joint_features)
        joint_features = self.feature_model_final_linear(joint_features)

        conv_output = torch.zeros([scatter_out_size, self.emb_size]).cuda().index_add(0, edge_indices[scatter_dim], joint_features)
        conv_output = self.post_conv_module(conv_output)

        output = torch.cat((conv_output, prev_features), dim=1)
        del conv_output
        output = self.output_module_layer_1(output)
        output = nn.functional.relu(output)
        output = self.output_module_layer_2(output)

        return output
    

