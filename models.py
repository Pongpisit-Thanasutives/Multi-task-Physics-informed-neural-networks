import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable, grad
from complexPyTorch.complexLayers import ComplexBatchNorm1d, ComplexDropout, ComplexLinear
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d
# from utils import *

from cplxmodule import nn as cnn
# complex valued tensor class
from cplxmodule import cplx
# converters
from cplxmodule.nn import RealToCplx, CplxToReal
# layers of encapsulating other complex valued layers
from cplxmodule.nn import CplxSequential
# common layers
from cplxmodule.nn import CplxLinear
# activation layers
from cplxmodule.nn import CplxModReLU

def cat(v1, v2): return torch.cat([v1, v2], dim=-1)

def cplx2tensor(func):
    return func.real + 1j*func.imag

def add_imaginary_dimension(a_tensor):
    return torch.hstack([a_tensor, torch.zeros(a_tensor.shape[0], 1).requires_grad_(False)])

def complex_mse(v1, v2):
    return F.mse_loss(v1.real, v2.real)+F.mse_loss(v1.imag, v2.imag)

def diff(func, inp):
    return grad(func, inp, create_graph=True, retain_graph=True, allow_unused=True, grad_outputs=torch.ones(func.shape, dtype=func.dtype))[0]

class TorchMLP(nn.Module):
    def __init__(self, dimensions, bias=True, activation_function=nn.Tanh(), bn=None, dropout=None):
        super(TorchMLP, self).__init__()
        self.model  = nn.ModuleList()

        for i in range(len(dimensions)-1):
            self.model.append(nn.Linear(dimensions[i], dimensions[i+1], bias=bias))
            if bn is not None and i!=len(dimensions)-2:
                self.model.append(bn(dimensions[i+1]))
                if dropout is not None:
                    self.model.append(dropout)
            if i==len(dimensions)-2: break
            self.model.append(activation_function)

    def xavier_init(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        for i, l in enumerate(self.model): 
            x = l(x)
        return x

class TorchComplexMLP(nn.Module):
    def __init__(self, dimensions, bias=True, activation_function=nn.Tanh(), bn=None, dropout_rate=0.0):
        super(TorchComplexMLP, self).__init__()
        self.model  = nn.ModuleList()
        self.dropout = None
        if dropout_rate>0.0: 
            self.dropout = ComplexDropout(dropout_rate)
        self.bn = bn

        for i in range(len(dimensions)-1):
            linear = ComplexLinear(dimensions[i], dimensions[i+1])
            linear.fc_r.apply(self.xavier_init)
            linear.fc_i.apply(self.xavier_init)
            self.model.append(linear)
            if self.bn is not None and i!=len(dimensions)-2:
                self.model.append(self.bn(dimensions[i+1]))
                if self.dropout is not None:
                    self.model.append(self.dropout)
            if i==len(dimensions)-2: break
            self.model.append(activation_function)

        try:
            self.model.apply(self.xavier_init)
        except:
            print("Cannot init the complex networ")
            pass

    def xavier_init(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        for i, l in enumerate(self.model): 
            x = l(x)
        return x
