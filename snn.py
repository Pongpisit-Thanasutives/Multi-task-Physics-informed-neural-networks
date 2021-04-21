import torch
import torch.nn as nn
import math
from collections import OrderedDict


class SNN(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim, n_layers, dropout_prob=0.0):
        super().__init__()
        layers = OrderedDict()
        for i in range(n_layers - 1):
            if i == 0:
                layers[f"fc{i}"] = nn.Linear(in_dim, hidden_dim, bias=False)
            else:
                layers[f"fc{i}"] = nn.Linear(hidden_dim, hidden_dim, bias=False)
            layers[f"selu_{i}"] = nn.SELU()
            layers[f"dropout_{i}"] = nn.AlphaDropout(p=dropout_prob)
        layers[f"fc_{i+1}"] = nn.Linear(hidden_dim, out_dim, bias=True)
        self.network = nn.Sequential(layers)
        self.reset_parameters()

    def forward(self, x):
        return self.network(x)

    def reset_parameters(self):
        for param in self.network.parameters():
            # biases zero
            if len(param.shape) == 1:
                nn.init.constant_(param, 0)
            # others using lecun-normal initialization
            else:
                nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='linear')

    def track_layer_activations(self, x):
        activations = []
        for layer in self.network:
            x = layer.forward(x)
            if isinstance(layer, nn.SELU):
                activations.append(x.data.flatten())
        return activations
