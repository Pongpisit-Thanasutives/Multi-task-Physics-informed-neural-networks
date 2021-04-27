from nupic.torch.modules import KWinners, SparseWeights, Flatten, rezero_weights, update_boost_strength

import torch
import torch.nn as nn
import torch.nn.functional as F

class TorchSparseMLP(nn.Module):
    def __init__(self, dimensions, sparsity=0.9, bias=True, is_kwinners=False, activation_function=nn.Tanh, bn=None, dropout=None, inp_drop=False, final_activation=None):
        super(TorchSparseMLP, self).__init__()
        self.model  = nn.ModuleList()
        # Can I also use the LayerNorm with elementwise_affine=True
        # This should be a callable module.
        self.is_kwinners = is_kwinners
        self.activation_function = activation_function()
        self.bn = bn
        if dropout is not None and inp_drop: self.inp_dropout = dropout
        else: self.inp_dropout = None
        self.model.apply(self.xavier_init)

        for i in range(len(dimensions)-1):
            if i==len(dimensions)-2:
                self.model.append(nn.Linear(dimensions[i], dimensions[i+1], bias=bias))
                break

            self.model.append(SparseWeights(nn.Linear(dimensions[i], dimensions[i+1], bias=bias), sparsity=sparsity))

            if self.is_kwinners:
                self.model.append(KWinners(dimensions[i+1], percent_on=0.1, boost_strength=1.0))
                
            if self.bn is not None:
                self.model.append(self.bn(dimensions[i+1]))
                if dropout is not None:
                    self.model.append(dropout)

            if self.activation_function is not None and not self.is_kwinners:
                self.model.append(activation_function())

        if final_activation is not None:
            self.model.append(final_activation())

    def xavier_init(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        if hasattr(self, 'inp_dropout'):
            if self.inp_dropout is not None:
                x = self.inp_dropout(x)
        for i, l in enumerate(self.model):
            x = l(x)
        return x
