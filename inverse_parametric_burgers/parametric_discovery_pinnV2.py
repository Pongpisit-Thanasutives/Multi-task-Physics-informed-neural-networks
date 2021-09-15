import sys; sys.path.insert(0, "../"); from utils import *
from models import SympyTorch, PartialDerivativeCalculator, CoeffLearner, UncertaintyWeightedLoss
import torch
from torch import nn
import torch.nn.functional as F

class ParametricSolver(nn.Module):
    def __init__(self, inp_dims=2, hidden_dims=50, out_dims=1, activation_module=nn.Tanh(), scale=False, lb=None, ub=None, input_feature=None):
        super(ParametricSolver, self).__init__()
        self.inp_dims = inp_dims
        self.hidden_dims = hidden_dims
        self.out_dims = out_dims
        self.activation_module = activation_module

        self.scale = scale
        self.lb = lb
        self.ub = ub

        self.model = nn.Sequential(nn.Linear(inp_dims, hidden_dims), self.activation_module, 
                                            nn.Linear(hidden_dims, hidden_dims), self.activation_module,
                                            nn.Linear(hidden_dims, hidden_dims), self.activation_module,
                                            nn.Linear(hidden_dims, hidden_dims), self.activation_module,
                                            nn.Linear(hidden_dims, hidden_dims), self.activation_module,
                                            nn.Linear(hidden_dims, out_dims))

        self.input_feature = input_feature

    def forward(self, x, t):
        inp = cat(x, t)
        if self.scale: inp = self.model(inp)                
        return self.model(inp)

    # This func should depend on the self.eq_name as well.
    def gradients_dict(self, x, t):
        u = self.forward(x, t)
        u_t = diff(u, t)
        u_x = diff(u, x)
        u_xx = diff(u_x, x)
        u_xxx = diff(u_xx, x)
        return cat(u, eval(self.input_feature), u_x, u_xx, u_xxx), u_t
    
    def neural_net_scale(self, inp):
        return -1.0 + 2.0*(inp-self.lb)/(self.ub-self.lb)

class RobustPCANN(nn.Module):
    def __init__(self, beta=0.0, is_beta_trainable=True, inp_dims=2, hidden_dims=50):
        super(RobustPCANN, self).__init__()
        if is_beta_trainable: self.beta = nn.Parameter(data=torch.FloatTensor([beta]), requires_grad=True)
        else: self.beta = beta
        self.proj = nn.Sequential(nn.Linear(inp_dims, hidden_dims), nn.Tanh(), nn.Linear(hidden_dims, inp_dims))

    def forward(self, O, S, order="fro", normalize=True):
        corr = self.proj(S)
        if normalize: corr = corr / torch.norm(corr, p=order)
        return O - self.beta*corr
    
class FuncNet(nn.Module):
    def __init__(self, inp_dims=2, n_funcs=2, hidden_dims=50, activation_module=nn.Tanh()):
        super(FuncNet, self).__init__()
        self.inp_dims = inp_dims
        self.activation_module = activation_module
        self.preprocessor_net= nn.Linear(self.inp_dims, self.inp_dims)
        self.weight = nn.Parameter(data=torch.tensor(0.5), requires_grad=True)
        self.neural_net = nn.Sequential(nn.Linear(self.inp_dims, hidden_dims), self.activation_module, 
                                        nn.Linear(hidden_dims, n_funcs))

    def forward(self, X):
        if self.inp_dims > 1:
            features = self.preprocessor_net(X)
            features = cat(features[:, 0:1]*self.weight, features[:, 1:2]*(1-self.weight))
            features = self.activation_module(features)
        else: features = X

        features = self.neural_net(features)
        return features

class FinalParametricPINN(nn.Module):
    def __init__(self, model, pde_terms, func_terms, uncert=False, scale=False, lb=None, ub=None, trainable_one=True):
        super(FinalParametricPINN, self).__init__()
        self.model = model
        self.pdc = PartialDerivativeCalculator(pde_terms, func_terms, trainable_one=trainable_one)
        self.loss_weightor = None
        self.is_uncert = uncert
        self.loss_weightor = None
        if self.is_uncert: self.loss_weightor = UncertaintyWeightedLoss(2)
        self.scale = scale
        self.lb = lb
        self.ub = ub

    def forward(self, x, t):
        inp = cat(x, t)
        if self.scale: inp = self.neural_net_scale(inp)
        return self.model(inp)

    def loss(self, x, t, y_train):
        u = self.forward(x, t)
        pde_loss = F.mse_loss(diff(u, t), self.pdc(u, x, t))
        mse_loss = F.mse_loss(u, y_train)
        if self.is_uncert: return self.loss_weightor(mse_loss, pde_loss)
        else: return [mse_loss, pde_loss]

    def neural_net_scale(self, inp):
        return -1.0 + 2.0*(inp-self.lb)/(self.ub-self.lb)

if __name__ == "__main__":
    model = ParametricPINN()
    print("Test init the model passed.")