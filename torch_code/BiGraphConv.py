# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 11:28:53 2020

@author: Manisha
"""
import torch
import torch.nn as nn
import tensorflow as tf

# Fully connected neural network with one hidden layer
from torch_code.pre_norm_layer import PreNormLayer, PreNormException
import torch.nn.functional as F

class BipartiteGraphConvolution(nn.Module):
    """
    Partial bipartite graph convolution (either left-to-right or right-to-left).
    Notes on Keras Dense layer to pytorch :
    *Dense layer in Keras implements output = activation(dot(input, kernel) + bias)
    *Linear layer + point-wise non-linearity/activation = Dense Layer in Keras
    *Kernel initialier = statistical ddistribution to intilize weights
    
    """
    def init_weights(x, initializer):
            if type(x) == nn.Linear :
                initializer(x.weight)
                #fill in weights, 0.1?
                x.weight.data.fill_(0.1)
                #add constant bias function which is just set to 0
                torch.nn.init.constant(x.bias.data, val=0)
                x.bias.data.fill_(0)
                
    def __init__(self, emb_size, activation, initializer, right_to_left=False, weights):
        
        super().__init__()
        
        self.emb_size = emb_size
        self.activation = activation
        self.initializer = initializer
        self.right_to_left = right_to_left
        
        self.weights = init_weights(x,initializer)
                
 
        # feature layers
        #self.feature_module_left = nn.Sequential([
        #    K.layers.Dense(units=self.emb_size, activation=None, use_bias=True, kernel_initializer=self.initializer)
        #])
        
        self.feature_module_left_1 = nn.Linear(input_shapes, self.emb_size)
        self.feature_module_left_2 = nn.Linear(input_shapes, self.emb_size)

        #self.feature_module_edge = K.Sequential([
        #    K.layers.Dense(units=self.emb_size, activation=None, use_bias=False, kernel_initializer=self.initializer)
        #])
        
        self.feature_module_edge = nn.Linear(input_shapes, self.emb_size)
        
        #self.feature_module_right = K.Sequential([
        #    K.layers.Dense(units=self.emb_size, activation=None, use_bias=False, kernel_initializer=self.initializer)
        #])
        
        self.feature_module_right_1 = nn.Linear(input_shapes, self.emb_size)
        self.feature_module_right_2 = nn.Linear(input_shapes, self.emb_size)

        
        self.feature_module_final = K.Sequential([
            PreNormLayer(1, shift=False),  # normalize after summation trick
            K.layers.Activation(self.activation),
            K.layers.Dense(units=self.emb_size, activation=None, kernel_initializer=self.initializer)
        ])

        self.post_conv_module = K.Sequential([
            PreNormLayer(1, shift=False),  # normalize after convolution
        ])

        # output_layers
        self.output_module = K.Sequential([
            K.layers.Dense(units=self.emb_size, activation=None, kernel_initializer=self.initializer),
            K.layers.Activation(self.activation),
            K.layers.Dense(units=self.emb_size, activation=None, kernel_initializer=self.initializer),
        ])
    
    def forward_c2v(self, l_shape, ev_shape, x_edge, activation, initializer):
        '''
        forward pass from constraints to variables
        
        cons_embedding -> feature_module_left_1
        
        feature_module_left_1 -> feature_module_left_2
        edge_embedding -> feature_module_left_2
        
        feature_model_left_2 -> feature_module_right_2
        '''
        x_left = self.feature_module_left_1(input_shapes=l_shape)
        x_edge = self.feature_module_edge(input_shapes=ev_shape)
        
        x = torch.cat((x_left,x_edge), dim = 1)
        
        x = self.activation(self.feature_module_left_2) 
        x = self.activation(self.feature_module_right_2)
        return x
    
    def forward_v2c():
        '''
        forward pass from variables to constraints
        vars_embedding -> feature_module_right_1
        feature_module_right_1 -> feature_module_right_2
        edge_embedding -> feature_module_right_2
        feature_model_left_2 -> feature_module_right_2
        '''
        return 1

        
    def build(self, input_shapes):
        l_shape, ei_shape, ev_shape, r_shape = input_shapes

        self.feature_module_left.build(l_shape)
        self.feature_module_edge.build(ev_shape)
        self.feature_module_right.build(r_shape)
        self.feature_module_final.build([None, self.emb_size])
        self.post_conv_module.build([None, self.emb_size])
        self.output_module.build([None, self.emb_size + (l_shape[1] if self.right_to_left else r_shape[1])])
        self.built = True

    def call(self, inputs, training):
        """
        Perfoms a partial graph convolution on the given bipartite graph.

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
        else:
            scatter_dim = 1
            prev_features = right_features

        # compute joint features
        joint_features = self.feature_module_final(
            tf.gather(
                self.feature_module_left(left_features),
                axis=0,
                indices=edge_indices[0]
            ) +
            self.feature_module_edge(edge_features) +
            tf.gather(
                self.feature_module_right(right_features),
                axis=0,
                indices=edge_indices[1])
        )

        # perform convolution
        conv_output = tf.scatter_nd(
            updates=joint_features,
            indices=tf.expand_dims(edge_indices[scatter_dim], axis=1),
            shape=[scatter_out_size, self.emb_size]
        )
        conv_output = self.post_conv_module(conv_output)

        # apply final module
        output = self.output_module(tf.concat([
            conv_output,
            prev_features,
        ], axis=1))

        return output
    

