#
# Author: Charles L. Bérubé
# Created on: Fri Jun 02 2023
#
# Copyright (c) 2023 C.L. Bérubé & J.-L. Gagnon
#


import torch
import torch.nn as nn


class MLP(nn.Module):
    # Réseau de neurones simple
    def __init__(
        self, input_dim, hidden_dim, output_dim, n_hidden, activation=nn.SiLU()
    ):
        super(MLP, self).__init__()

        # Hyper-paramètres du modèle
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_hidden = n_hidden
        self.activation = activation

        # Définition des couches
        layer_list = [nn.Linear(input_dim, hidden_dim)]
        layer_list.extend(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden - 1)]
        )
        layer_list.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.ModuleList(layer_list)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i < self.n_hidden:
                # Activation function applied to each hidden layer
                x = self.activation(layer(x))
            if i == self.n_hidden:
                # Squash the output to [0, 1] with a sigmoid (improvement from the paper)
                x = torch.sigmoid(layer(x))
        return x
