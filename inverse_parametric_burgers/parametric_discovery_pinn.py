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
        if self.n_funcs == 0: print("You are not using the parametric_func_net")
        self.activation_module = activation_module
        self.scale = scale
        self.lb = lb
        self.ub = ub

        self.preprocessor_net = nn.Sequential(nn.Linear(inp_dims, hidden_dims), self.activation_module, 
                                              nn.Linear(hidden_dims, hidden_dims), self.activation_module)

        self.parametric_func_net = None
        if self.n_funcs>0:
            # I should change from 1 -> 2 for being more real.
#           self.parametric_func_net = nn.Sequential(nn.Linear(2, hidden_dims), self.activation_module,
#                                                    nn.Linear(hidden_dims, self.n_funcs))

            self.parametric_func_net = FuncNet(inp_dims=2, n_funcs=self.n_funcs, hidden_dims=hidden_dims,
                                                activation_module=self.activation_module)

        self.pde_solver_net = nn.Sequential(nn.Linear(hidden_dims, hidden_dims), self.activation_module,
                                            nn.Linear(hidden_dims, hidden_dims), self.activation_module,
                                            nn.Linear(hidden_dims, hidden_dims), self.activation_module,
                                            nn.Linear(hidden_dims, out_dims)) 

    def forward(self, x, t):
        inp = cat(x, t)
        if self.scale: inp = self.neural_net_scale(inp)

        features = self.preprocessor_net(inp)

        learned_funcs = None
        if self.n_funcs>0: learned_funcs = self.parametric_func_net(inp)
                
        u = self.pde_solver_net(features)

        return u, learned_funcs 

    def gradients_dict(self, x, t):
        inp = cat(x, t)
        if self.scale: inp = self.neural_net_scale(inp)
        features = self.preprocessor_net(inp)
        u = self.pde_solver_net(features)
        u_t = diff(u, t)
        u_x = diff(u, x)
        u_xx = diff(u_x, x)
        return {'u':u, 'u_t':u_t, 'u_x':u_x, 'u_xx':u_xx}

    def loss(self, x, t, y_train):
        inp = cat(x, t)
        if self.scale: inp = self.neural_net_scale(inp)

        features = self.preprocessor_net(inp)

        learned_funcs = None
        if self.n_funcs>0: learned_funcs = self.parametric_func_net(inp)
        else: pde_loss = 0.0

        # Change this part of the codes for discovering different PDEs
        u = self.pde_solver_net(features)
        u_t = diff(u, t)
        u_x = diff(u, x)
        u_xx = diff(u_x, x)

        if learned_funcs is not None: pde_loss = F.mse_loss(learned_funcs[:, 0:1]*u*u_x + learned_funcs[:, 1:2]*u_xx, u_t)
        mse_loss = F.mse_loss(u, y_train)

        return mse_loss, pde_loss
    
    def neural_net_scale(self, inp):
        return -1.0 + 2.0*(inp-self.lb)/(self.ub-self.lb)

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

# Parent class
class FinalPINN(nn.Module):
    def __init__(self, model, funcs, scale=False, lb=None, ub=None):
        super(FinalPINN, self).__init__()
        self.model = model
        self.funcs = nn.ModuleList(funcs)
        self.scale = scale
        self.lb = lb
        self.ub = ub

    def forward(self, x, t):
        inp = cat(x, t)
        if self.scale: inp = self.neural_net_scale(inp)
        return self.model(inp)

    @staticmethod
    def loss(self, x, t, y_train):
        pass

    def neural_net_scale(self, inp):
        return -1.0 + 2.0*(inp-self.lb)/(self.ub-self.lb)

# Only for discovering the parametric Burgers' equation
# No other use
class BurgerPINN(nn.Module):
    def __init__(self, model, funcs, epsilon=0.09875935, scale=False, lb=None, ub=None):
        super(BurgerPINN, self).__init__()
        self.model = model
        self.funcs = funcs 
        self.epsilon = nn.Parameter(torch.tensor(epsilon), requires_grad=True)
        self.scale = scale
        self.lb = lb
        self.ub = ub

    def forward(self, x, t):
        inp = cat(x, t)
        if self.scale: inp = self.neural_net_scale(inp)
        return self.model(inp)

    def loss(self, x, t, y_train):
        u = self.forward(x, t)

        u_t = diff(u, t)
        u_x = diff(u, x)
        u_xx = diff(u_x, x)

        pde_loss = F.mse_loss(torch.squeeze(self.funcs(t=t), -1)*u*u_x+self.epsilon*u_xx, u_t)
        mse_loss = F.mse_loss(u, y_train)

        return mse_loss, pde_loss

    def neural_net_scale(self, inp):
        return -1.0 + 2.0*(inp-self.lb)/(self.ub-self.lb)

if __name__ == "__main__":
    model = ParametricPINN()
    print("Test init the model passed.")
