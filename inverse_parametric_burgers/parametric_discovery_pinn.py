import sys; sys.path.insert(0, "../"); from utils import *
import torch
from torch import nn
import torch.nn.functional as F

class ParametricPINN(nn.Module):
    def __init__(self, inp_dims=2, hidden_dims=50, out_dims=1, activation_module=nn.Tanh(), n_funcs=2, scale=False, lb=None, ub=None):
        super(ParametricPINN, self).__init__()
        # The default config is only for the Burgers' equation.
        self.inp_dims = inp_dims    
        self.hidden_dims = hidden_dims
        self.out_dims = out_dims
        self.n_funcs = n_funcs
        self.activation_module = activation_module
        self.scale = scale
        self.lb = lb
        self.ub = ub

        self.preprocessor_net = nn.Sequential(nn.Linear(inp_dims, hidden_dims), self.activation_module, 
                                              nn.Linear(hidden_dims, hidden_dims), self.activation_module)

        self.parametric_func_net = None
        if self.n_funcs>0:
            # I should change from 1 -> 2 for being more real.
            self.parametric_func_net = nn.Sequential(nn.Linear(1, hidden_dims), self.activation_module,
                                                     nn.Linear(hidden_dims, self.n_funcs))

        self.pde_solver_net = nn.Sequential(nn.Linear(hidden_dims, hidden_dims), self.activation_module,
                                            nn.Linear(hidden_dims, hidden_dims), self.activation_module,
                                            nn.Linear(hidden_dims, hidden_dims), self.activation_module,
                                            nn.Linear(hidden_dims, out_dims)) 

    def forward(self, x, t):
        inp = cat(x, t)
        if self.scale: inp = self.neural_net_scale(inp)

        features = self.preprocessor_net(inp)
        learned_funcs = self.parametric_func_net(inp[:, 1:2])
        u = self.pde_solver_net(features)

        return u, learned_funcs 

    def loss(self, x, t, y_train):
        inp = cat(x, t)
        if self.scale: inp = self.neural_net_scale(inp)

        features = self.preprocessor_net(inp)
        learned_funcs = self.parametric_func_net(inp[:, 1:2])

        # Change this part of the codes for discovering different PDEs
        u = self.pde_solver_net(features)
        u_t = diff(u, t)
        u_x = diff(u, x)
        u_xx = diff(u_x, x)

        pde_loss = F.mse_loss(learned_funcs[:, 0:1]*u*u_x + learned_funcs[:, 1:2]*u_xx, u_t)
        mse_loss = F.mse_loss(u, y_train)

        return mse_loss, pde_loss
    
    def neural_net_scale(self, inp):
        return -1.0 + 2.0*(inp-self.lb)/(self.ub-self.lb)

if __name__ == "__main__":
    model = ParametricPINN()
    print("Test init the model passed.")
