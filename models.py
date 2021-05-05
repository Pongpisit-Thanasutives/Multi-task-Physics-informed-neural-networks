import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable, grad
from complexPyTorch.complexLayers import ComplexBatchNorm1d, ComplexDropout, ComplexLinear
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d

from cplxmodule import nn as cnn
from cplxmodule import cplx
from cplxmodule.nn import RealToCplx, CplxToReal
from cplxmodule.nn import CplxSequential, CplxLinear, CplxModReLU

def cat(*args): return torch.cat(args, dim=-1)

def cplx2tensor(func):
    return func.real + 1j*func.imag

def add_imaginary_dimension(a_tensor):
    return torch.hstack([a_tensor, torch.zeros(a_tensor.shape[0], 1).requires_grad_(False)])

def complex_mse(v1, v2):
    return F.mse_loss(v1.real, v2.real)+F.mse_loss(v1.imag, v2.imag)

def diff(func, inp):
    return grad(func, inp, create_graph=True, retain_graph=True, allow_unused=True, grad_outputs=torch.ones(func.shape, dtype=func.dtype))[0]

class ImaginaryDimensionAdder(nn.Module):
    def __init__(self,):
        super(ImaginaryDimensionAdder, self).__init__(); pass
    def forward(self, real_tensor):
        added = cat(real_tensor[:, 0:1], torch.zeros(real_tensor.shape[0], 1))
        for i in range(1, real_tensor.shape[1]):
            added = cat(added, real_tensor[:, i:i+1], torch.zeros(real_tensor.shape[0], 1))
        return added

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

class Network(nn.Module):
    def __init__(self, model, index2features = ('uf', 'u_x',  'u_xx', 'u_tt', 'u_xt', 'u_tx')):
        super(Network, self).__init__()
        # pls init the self.model before
        self.model = model
        # For tracking
        self.index2features = index2features 
        self.uf = None

    def xavier_init(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x, t):
        self.uf = self.model(torch.cat([x, t], dim=1))
        return self.uf

    def get_selector_data(self, x, t):
        uf = self.forward(x, t)

        ### PDE Loss calculation ###
        # first-order derivatives
        u_t = self.gradients(uf, t)[0]
        u_x = self.gradients(uf, x)[0]
        # Homo second-order derivatives
        u_tt = self.gradients(u_t,t)[0]
        u_xx = self.gradients(u_x, x)[0]
        # Hetero second-order derivatives
        u_xt = self.gradients(u_t, x)[0]
        u_tx = self.gradients(u_x, t)[0]

        X_selector = torch.cat([uf, u_x, u_xx, u_tt, u_xt, u_tx], dim=1)
        y_selector = u_t

        return X_selector, y_selector

    def gradients(self, func, x):
        return grad(func, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones(func.shape))

class SeclectorNetwork(nn.Module):
    def __init__(self, X_train_dim, bn=None):
        super(SeclectorNetwork, self).__init__()
        # Nonlinear model, Training with PDE reg.
        self.nonlinear_model = TorchMLP(dimensions=[X_train_dim, 50, 50, 1], activation_function=nn.Tanh(), bn=bn, dropout=nn.Dropout(p=0.1))
        
    def xavier_init(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        
    def forward(self, inn):
        ut_approx = self.nonlinear_model(inn)
        return ut_approx
    
    def loss(self, X_input, y_input):
        ut_approx = self.forward(X_input)
        mse_loss = F.mse_loss(ut_approx, y_input, reduction='mean')
        return mse_loss

class SemiSupModel(nn.Module):
    def __init__(self, network, selector, normalize_derivative_features=False, mini=None, maxi=None):
        super(SemiSupModel, self).__init__()
        self.network = network
        self.selector = selector
        self.normalize_derivative_features = normalize_derivative_features
        self.mini = mini
        self.maxi = maxi
    def forward(self, X_u_train):
        X_selector, y_selector = self.network.get_selector_data(*dimension_slicing(X_u_train))
        if self.normalize_derivative_features:
            X_selector = (X_selector-self.mini)/(self.maxi-self.mini)
        unsup_loss = self.selector.loss(X_selector, y_selector)
        return self.network.uf, unsup_loss
