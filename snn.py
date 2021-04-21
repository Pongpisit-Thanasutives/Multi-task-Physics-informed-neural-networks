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
        for layer in self.network:
            if not isinstance(layer, nn.Linear):
                continue
            nn.init.normal_(layer.weight, std=1 / math.sqrt(layer.out_features))
            if layer.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(layer.bias, -bound, bound)

    def track_layer_activations(self, x):
        activations = []
        for layer in self.network:
            x = layer.forward(x)
            if isinstance(layer, nn.SELU):
                activations.append(x.data.flatten())
        return activations
