#!/usr/bin/env python
import matplotlib.pyplot as plt

import numpy as np
import scipy.io as io
from pyDOE import lhs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from complexPyTorch.complexLayers import ComplexLinear

import cplxmodule
from cplxmodule import cplx
from cplxmodule.nn import RealToCplx, CplxToReal, CplxSequential, CplxToCplx
from cplxmodule.nn import CplxLinear, CplxModReLU, CplxAdaptiveModReLU, CplxModulus, CplxAngle

# To access the contents of the parent dir
import sys; sys.path.insert(0, '../')
import os
from scipy.io import loadmat
from lightning_utils import *
from utils import *
from models import (TorchComplexMLP, ImaginaryDimensionAdder, cplx2tensor, 
                    ComplexTorchMLP, ComplexSymPyModule, complex_mse)
from models import UncertaintyWeightedLoss
from preprocess import *

# Model selection
from sparsereg.model import STRidge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from pde_diff import TrainSTRidge, FiniteDiff, print_pde
from RegscorePy.bic import bic

from madgrad import MADGRAD


# In[2]:


# torch device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("You're running on", device)

DATA_PATH = '../PDE_FIND_experimental_datasets/harmonic_osc.mat'
data = io.loadmat(DATA_PATH)

t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]

spatial_dim = x.shape[0]
time_dim = t.shape[0]

potential = np.vstack([0.5*np.power(x,2).reshape((1, spatial_dim)) for _ in range(time_dim)])
Exact = data['usol']

X, T = np.meshgrid(x, t)

# Adjust the diemnsion of Exact and potential (0.5*x**2)
if Exact.T.shape == X.shape: Exact = Exact.T
if potential.T.shape == X.shape: potential = potential.T
Exact_u = np.real(Exact)
Exact_v = np.imag(Exact)

# Converting in a feature vector for each feature
X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
h_star = to_column_vector(Exact)
u_star = to_column_vector(Exact_u)
v_star = to_column_vector(Exact_v)
potential = to_column_vector(potential)

# Doman bounds
lb = X_star.min(axis=0)
ub = X_star.max(axis=0)

# Converting the grounds to be tensor
X_star = to_tensor(X_star, True)
h_star = to_complex_tensor(h_star, False)

N = 40000; include_N_res = 2
idx = np.random.choice(X_star.shape[0], N, replace=False)
# idx = np.arange(N) # Just have an easy dataset for experimenting

lb = to_tensor(lb, False).to(device)
ub = to_tensor(ub, False).to(device)

X_train = to_tensor(X_star[idx, :], True).to(device)
u_train = to_tensor(u_star[idx, :], False).to(device)
v_train = to_tensor(v_star[idx, :], False).to(device)
h_train = torch.complex(u_train, v_train).to(device)
potential = to_tensor(potential[idx, :], False).to(device)

# X_train = to_tensor(np.load("./tmp_files/X_train_2000labeledsamplesV3.npy")[:N, :], True).to(device)
# h_train = to_complex_tensor(np.load("./tmp_files/h_train_2000labeledsamplesV3.npy"), False).to(device)

# Unsup data -> We do not need this here.
# Potential is calculated from x
# Hence, Quadratic features of x are required.
feature_names = ['hf', 'h_xx', 'V']


# In[3]:


# Define the initial coeffs
# u_t = (-0.000084 +0.494702i)h_xx
#     + (0.001434 -0.994890i)hf V

cn1 = (-0.000084-0.994890*1j)
cn2 = (0.001434+0.494702*1j)
cns = [cn1, cn2]


# In[4]:


# Type the equation got from the symbolic regression step
# No need to save the eq save a pickle file before
program1 = "X0*X2"
pde_expr1, variables1,  = build_exp(program1); print(pde_expr1, variables1)

program2 = "X1"
pde_expr2, variables2,  = build_exp(program2); print(pde_expr2, variables2)

mod = ComplexSymPyModule(expressions=[pde_expr1, pde_expr2], complex_coeffs=cns, learnable_parts=[True, True]); mod.train()


# In[5]:


class ComplexPINN(nn.Module):
    def __init__(self, model, loss_fn, index2features, scale=False, lb=None, ub=None, uncert=False):
        super(ComplexPINN, self).__init__()
        self.model = model
        self.callable_loss_fn = loss_fn
        self.index2features = index2features; self.feature2index = {}
        for idx, fn in enumerate(self.index2features): self.feature2index[fn] = str(idx)
        self.scale = scale; self.lb, self.ub = lb, ub
        if self.scale and (self.lb is None or self.ub is None): 
            print("Please provide thw lower and upper bounds of your PDE.")
            print("Otherwise, there will be error(s)")
        self.diff_flag = diff_flag(self.index2features)
        self.uncert = None
        if uncert: self.uncert = UncertaintyWeightedLoss(2)
        
    def forward(self, x, t):
        H = torch.cat([x, t], dim=1)
        if self.scale: H = self.neural_net_scale(H)
        return self.model(H)
    
    def loss(self, x, t, potential, y_input, update_network_params=True, update_pde_params=True):
        total_loss = []
        grads_dict, u_t = self.grads_dict(x, t, potential)
        # MSE Loss
        if update_network_params:
            total_loss.append(complex_mse(grads_dict['X0'], y_input))
        # PDE Loss
        if update_pde_params:
            total_loss.append(complex_mse(self.callable_loss_fn(grads_dict), u_t))
            
        if self.uncert is not None: return self.uncert(*total_loss)
        else: return total_loss
    
    def grads_dict(self, x, t, potential):
        uf = self.forward(x, t)
        u_t = complex_diff(uf, t)
        
        ### PDE Loss calculation ###
        # Without calling grad
        derivatives = {}
        derivatives['X0'] = cplx2tensor(uf)
        derivatives['X1'] = complex_diff(complex_diff(uf, x), x)
        derivatives['X2'] = potential
                
        # With calling grad
#         for t in self.diff_flag[1]:
#             out = uf
#             for c in t:
#                 if c=='x': out = complex_diff(out, x)
#                 elif c=='t': out = complex_diff(out, t)
#             derivatives['X'+self.feature2index['h_'+t[::-1]]] = out
        
        return derivatives, u_t
    
    def gradients(self, func, x):
        return grad(func, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones(func.shape))
    
    # Must ensure that the implementation of neural_net_scale is consistent
    # and hopefully correct
    # also, you might not need this function in some datasets
    def neural_net_scale(self, inp): 
        return 2*(inp-self.lb)/(self.ub-self.lb)-1


# In[6]:


inp_dimension = 2
act = CplxToCplx[torch.tanh]
complex_model = CplxSequential(
                            CplxLinear(100, 100, bias=True),
                            act(),
                            CplxLinear(100, 100, bias=True),
                            act(),
                            CplxLinear(100, 100, bias=True),
                            act(),
                            CplxLinear(100, 100, bias=True),
                            act(),
                            CplxLinear(100, 1, bias=True),
                            )

complex_model = torch.nn.Sequential(
                                    torch.nn.Linear(inp_dimension, 200),
                                    RealToCplx(),
                                    complex_model
                                    )


# In[7]:


# Pretrained model
semisup_model_state_dict = cpu_load("./saved_path_inverse_qho/qho_complex_model_2000labeledsamples_jointtrainwith4000unlabeledsamplesV3.pth")
parameters = OrderedDict()

# Filter only the parts that I care about renaming (to be similar to what defined in TorchMLP).
inner_part = "network.model."
for p in semisup_model_state_dict:
    if inner_part in p:
        parameters[p.replace(inner_part, "")] = semisup_model_state_dict[p]
complex_model.load_state_dict(parameters)

pinn = ComplexPINN(model=complex_model, loss_fn=mod, index2features=feature_names, scale=True, lb=lb, ub=ub, uncert=False)
# pinn.load_state_dict(torch.load("tmp.pth"), strict=False)


# In[8]:


def closure():
    global X_train, h_train, potential
    if torch.is_grad_enabled(): optimizer2.zero_grad(set_to_none=True)
    losses = pinn.loss(X_train[:, 0:1], X_train[:, 1:2], potential, h_train, update_network_params=True, update_pde_params=True)
    l = sum(losses)
    if l.requires_grad: l.backward(retain_graph=True)
    return l

def mtl_closure():
    global X_train, h_train, potential
    n_obj = 2 # There are two tasks
    losses = pinn.loss(X_train[:, 0:1], X_train[:, 1:2], potential, h_train, update_network_params=True, update_pde_params=True)
    updated_grads = []
    
    for i in range(n_obj):
        optimizer1.zero_grad(set_to_none=True)
        losses[i].backward(retain_graph=True)

        g_task = []
        for param in pinn.parameters():
            if param.grad is not None:
                g_task.append(Variable(param.grad.clone(), requires_grad=False))
            else:
                g_task.append(Variable(torch.zeros(param.shape), requires_grad=False))
        # appending the gradients from each task
        updated_grads.append(g_task)

    updated_grads = list(pcgrad.pc_grad_update(updated_grads))[0]
    for idx, param in enumerate(pinn.parameters()): 
        param.grad = (updated_grads[0][idx]+updated_grads[1][idx])
        
    return sum(losses)


# In[9]:


epochs1, epochs2 = 200, 30
# TODO: Save best state dict and training for more epochs.
# optimizer1 = MADGRAD(pinn.parameters(), lr=1e-7, momentum=0.9)
# pinn.train(); best_train_loss = 1e6

# print('1st Phase optimization using Adam with PCGrad gradient modification')
# for i in range(epochs1):
#     optimizer1.step(mtl_closure)
#     l = mtl_closure()
#     if (i % 10) == 0 or i == epochs1-1:
#         print("Epoch {}: ".format(i), l.item())


# In[ ]:


optimizer2 = torch.optim.LBFGS(pinn.parameters(), lr=1e-1, max_iter=500, max_eval=int(500*1.25), history_size=300, line_search_fn='strong_wolfe')
print('2nd Phase optimization using LBFGS')
for i in range(epochs2):
    optimizer2.step(closure)
    if (i % 5) == 0 or i == epochs2-1:
       print(closure().item()) 

p1, p2 = see_params(pinn.callable_loss_fn)
print(torch.complex(p1, p2))

save(pinn, "tmp.pth")
