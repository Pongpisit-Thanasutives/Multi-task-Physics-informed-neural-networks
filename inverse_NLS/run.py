#!/usr/bin/env python
# coding: utf-8

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
from utils import *
from models import TorchComplexMLP, ImaginaryDimensionAdder, cplx2tensor, ComplexTorchMLP, complex_mse
from preprocess import *

# Model selection
from sparsereg.model import STRidge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from pde_diff import TrainSTRidge, FiniteDiff, print_pde
from RegscorePy.bic import bic

from madgrad import MADGRAD


# In[2]:


# from complexPyTorch.complexLayers import ComplexBatchNorm1d, ComplexLinear
# from complexPyTorch.complexFunctions import complex_relu
# class ComplexNet(nn.Module):
#     def __init__(self):
#         super(ComplexNet, self).__init__()
#         self.fc1 = ComplexLinear(5, 100)
#         self.fc2 = ComplexLinear(100, 100)
#         self.fc3 = ComplexLinear(100, 1)
#     def forward(self, inp):
#         inp = complex_relu(self.fc1(inp))
#         inp = complex_relu(self.fc2(inp))
#         return self.fc3(inp)


# In[3]:


# torch device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("You're running on", device)

# Doman bounds
lb = np.array([-5.0, 0.0])
ub = np.array([5.0, np.pi/2])

N = 15000

DATA_PATH = '../experimental_data/NLS.mat'
data = io.loadmat(DATA_PATH)

t = data['tt'].flatten()[:,None]
x = data['x'].flatten()[:,None]
Exact = data['uu']
Exact_u = np.real(Exact)
Exact_v = np.imag(Exact)

X, T = np.meshgrid(x,t)

X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
u_star = to_column_vector(Exact_u.T)
v_star = to_column_vector(Exact_v.T)

idx = np.random.choice(X_star.shape[0], N, replace=False)

lb = to_tensor(lb, False).to(device)
ub = to_tensor(ub, False).to(device)

X_train = to_tensor(X_star[idx, :], True).to(device)
u_train = to_tensor(u_star[idx, :], False).to(device)
v_train = to_tensor(v_star[idx, :], False).to(device)

feature_names = ['hf', '|hf|', 'h_x', 'h_xx', 'h_xxx']


# In[4]:


spatial_dim = x.shape[0]
time_dim = t.shape[0]

dt = (t[1]-t[0])[0]
dx = (x[2]-x[1])[0]

fd_h_t = np.zeros((spatial_dim, time_dim), dtype=np.complex64)
fd_h_x = np.zeros((spatial_dim, time_dim), dtype=np.complex64)
fd_h_xx = np.zeros((spatial_dim, time_dim), dtype=np.complex64)
fd_h_xxx = np.zeros((spatial_dim, time_dim), dtype=np.complex64)

for i in range(spatial_dim):
    fd_h_t[i,:] = FiniteDiff(Exact[i,:], dt, 1)
for i in range(time_dim):
    fd_h_x[:,i] = FiniteDiff(Exact[:,i], dx, 1)
    fd_h_xx[:,i] = FiniteDiff(Exact[:,i], dx, 2)
    fd_h_xxx[:,i] = FiniteDiff(Exact[:,i], dx, 3)
    
fd_h_t = np.reshape(fd_h_t, (spatial_dim*time_dim,1), order='F')
fd_h_x = np.reshape(fd_h_x, (spatial_dim*time_dim,1), order='F')
fd_h_xx = np.reshape(fd_h_xx, (spatial_dim*time_dim,1), order='F')
fd_h_xxx = np.reshape(fd_h_xxx, (spatial_dim*time_dim,1), order='F')


# In[5]:


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

complex_model.load_state_dict(cpu_load("./saved_path_inverse_nls/NLS_cpinn_model.pth"))


# In[6]:


class ComplexPhysicsInformedNN(nn.Module):
    def __init__(self, model, lb, ub, scale=False):
        super(ComplexPhysicsInformedNN, self).__init__()
        self.model = model
        self.lb = lb
        self.ub = ub
        self.scale = scale
    
    def forward(self, X):
        if self.scale: 
            return self.model(self.neural_net_scale(X))
        return self.model(X)

    def predict(self, X_test):
        return CplxToReal()(self.forward(self.preprocess(*dimension_slicing(X_test))))
    
    def neural_net_scale(self, inp):
        return (2.0*(inp-self.lb)/(self.ub-self.lb))-1.0

    def preprocess(self, spatial, time):
        return cat(spatial, time)
    
    def loss(self, X_f, X0, h0, X_lb, X_ub):
        loss = self.net_f(*dimension_slicing(X_f))
        h0_pred = self.predict(X0); u0 = h0_pred[:, 0:1]; v0 = h0_pred[:, 1:2]
        loss += F.mse_loss(u0, h0[:, 0:1])+F.mse_loss(v0, h0[:, 1:2])
        u_lb, v_lb, u_lb_x, v_lb_x = self.net_h(*dimension_slicing(X_lb))
        u_ub, v_ub, u_ub_x, v_ub_x = self.net_h(*dimension_slicing(X_ub))
        loss += F.mse_loss(u_lb, u_ub)
        loss += F.mse_loss(v_lb, v_ub)
        loss += F.mse_loss(u_lb_x, u_ub_x)
        loss += F.mse_loss(v_lb_x, v_ub_x)
        return loss
    
    def net_h(self, x, t):
        X = cat(x, t)
        h = self.forward(X)
        u = h.real
        v = h.imag
        return u, v, self.diff(u, x), self.diff(v, x)
    
    def net_f(self, x, t):
        u, v, u_x, v_x = self.net_h(x, t)
        u_t, v_t = self.diff(u, t), self.diff(v, t)
        u_xx, v_xx = self.diff(u_x, x), self.diff(v_x, x)
        f_u = u_t + 0.5*v_xx + (u**2 + v**2)*v
        f_v = v_t - 0.5*u_xx - (u**2 + v**2)*u
        return (f_u**2).mean()+(f_v**2).mean()

    def diff(self, func, inp):
        return grad(func, inp, create_graph=True, retain_graph=True, grad_outputs=torch.ones(func.shape, dtype=func.dtype).to(device))[0]

    def complex_mse(self, v1, v2):
        assert v1.shape == v2.shape
        assert v1.shape[1] == 1
        return F.mse_loss(v1.real, v2.real)+F.mse_loss(v2.imag, v2.imag)

    def add_imag_dim(self, v1):
        z = torch.zeros(v1.shape).requires_grad_(False).to(device)
        return torch.complex(v1, z)
    
cpinn = ComplexPhysicsInformedNN(model=complex_model, lb=lb, ub=ub, scale=False).to(device)
cpinn.load_state_dict(cpu_load("./saved_path_inverse_nls/NLS_cpinn.pth"))


# #### Goals
# (1) Re-implement the semisup_model for a complex network.
# 
# (2) Implement the self.gradients function.
# - complex_model(input) -> diff(u_pred, x) & diff(v_pred, x) -> combine 2 diff terms as 1 complex vector -> compute PDE loss / passing to the selector network

# ### some tests

# In[7]:


# xx, tt = dimension_slicing(to_tensor(X_train, True))
# predictions = complex_model(cat(xx, tt))
# h = cplx2tensor(predictions)
# h_x = complex_diff(predictions, xx)
# h_xx = complex_diff(h_x, xx)
# h_xxx = complex_diff(h_xx, xx)
# h_t = complex_diff(predictions, tt)


# In[8]:


# f = 1j*h_t+0.5*h_xx+(h.abs()**2)*h


# In[9]:


# real_loss = (f.real**2).mean(); imag_loss = (f.imag**2).mean()
# avg_loss = (real_loss+imag_loss)*0.5
# print("PDE Loss", avg_loss.item())
# print("MSE Loss", complex_mse(predictions, u_train+1j*v_train).item())


# In[10]:


# derivatives = to_numpy(cat(h, h.abs()**2, h_x, h_xx, h_xxx))
# dictionary = {}
# for i in range(len(feature_names)): dictionary[feature_names[i]] = get_feature(derivatives, i)
# dictionary


# In[11]:


# c_poly = ComplexPolynomialFeatures(feature_names, dictionary)
# complex_poly_features = c_poly.fit()
# complex_poly_features


# In[12]:


# w = TrainSTRidge(complex_poly_features, to_numpy(h_t), 1e-6, 1000, maxit=100)
# print("PDE derived using STRidge")
# print_pde(w, c_poly.poly_feature_names)


# #### Automatic differentiation w/ and w/o Finite difference guidance

# In[13]:


class ComplexNetwork(nn.Module):
    def __init__(self, model, index2features=None, scale=False, lb=None, ub=None):
        super(ComplexNetwork, self).__init__()
        # pls init the self.model before
        self.model = model
        # For tracking, the default tup is for the burgers' equation.
        self.index2features = index2features
        print("Considering", self.index2features)
        self.diff_flag = diff_flag(self.index2features)
        self.uf = None
        self.scale = scale
        self.lb, self.ub = lb, ub
        
    def xavier_init(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x, t):
        if not self.scale: self.uf = self.model(torch.cat([x, t], dim=1))
        else: self.uf = self.model(self.neural_net_scale(torch.cat([x, t], dim=1)))
        return self.uf
    
    def get_selector_data(self, x, t):
        uf = self.forward(x, t)
        u_t = complex_diff(uf, t)
        
        ### PDE Loss calculation ###
        # Without calling grad
        derivatives = []
        for t in self.diff_flag[0]:
            if t=='hf': 
                derivatives.append(cplx2tensor(uf))
                derivatives.append((uf.real**2+uf.imag**2)+0.0j)
            elif t=='x': derivatives.append(x)
        # With calling grad
        for t in self.diff_flag[1]:
            out = uf
            for c in t:
                if c=='x': out = complex_diff(out, x)
                elif c=='t': out = complex_diff(out, t)
            derivatives.append(out)
        
        return torch.cat(derivatives, dim=-1), u_t
    
    def neural_net_scale(self, inp):
        return 2*(inp-self.lb)/(self.ub-self.lb)-1


# In[14]:


complex_network = ComplexNetwork(model=complex_model, index2features=feature_names, scale=True, lb=lb, ub=ub)
X_selector, y_selector = complex_network.get_selector_data(*dimension_slicing(X_train))


# In[15]:


class ComplexAttentionSelectorNetwork(nn.Module):
    def __init__(self, layers, prob_activation=torch.sigmoid, bn=None, reg_intensity=0.1):
        super(ComplexAttentionSelectorNetwork, self).__init__()
        # Nonlinear model, Training with PDE reg.
        assert len(layers) > 1
        self.linear1 = CplxLinear(layers[0], layers[0], bias=True)
        self.prob_activation = prob_activation
        self.nonlinear_model = ComplexTorchMLP(dimensions=layers, activation_function=CplxToCplx[F.relu](), bn=bn, dropout_rate=0.0)
        self.latest_weighted_features = None
        self.th = 0.1
        self.reg_intensity = reg_intensity
        
    def xavier_init(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        
    def forward(self, inn):
        feature_importances = self.weighted_features(inn)
        inn = inn*feature_importances
        return self.nonlinear_model(inn)
    
    def weighted_features(self, inn):
        self.latest_weighted_features = self.prob_activation(cplx2tensor(self.linear1(inn)).abs())
        self.latest_weighted_features = self.latest_weighted_features.mean(dim=0)
        return self.latest_weighted_features
    
    def loss(self, X_input, y_input):
        ut_approx = self.forward(X_input)
        mse_loss = complex_mse(ut_approx, y_input)
        reg_term = F.relu(self.latest_weighted_features-self.th)
        return mse_loss+self.reg_intensity*(torch.norm(reg_term, p=0)+(torch.tensor([1.0, 1.0, 1.0, 2.0, 3.0])*reg_term).sum())

# Only the SemiSupModel has changed to work with the finite difference guidance
class SemiSupModel(nn.Module):
    def __init__(self, network, selector, normalize_derivative_features=False, mini=None, maxi=None):
        super(SemiSupModel, self).__init__()
        self.network = network
        self.selector = selector
        self.normalize_derivative_features = normalize_derivative_features
        self.mini = mini
        self.maxi = maxi
        
    def forward(self, X_u_train, fd_derivatives=None, fd_u_t=None, fd_weight=0.0, include_unsup=True):
        X_selector, y_selector = self.network.get_selector_data(*dimension_slicing(X_u_train))
        
        print(X_selector)
        
        fd_guidance = 0.0
        if fd_weight>0.0 and fd_derivatives is not None and fd_u_t is not None:
            # Traditional MSE Loss btw uf and u_train + the fd_guidance loss
            row, col = fd_derivatives.shape
            fd_guidance += complex_mse(X_selector[:row, 0:1], fd_derivatives[:, 0:1])
            fd_guidance += fd_weight*(col-1)*complex_mse(X_selector[:row, 1:], fd_derivatives[:, 1:])
            fd_guidance += fd_weight*complex_mse(y_selector[:row, :], fd_u_t)
            
        else: fd_guidance = self.network.uf
            
        # I am not sure a good way to normalize/scale a complex tensor
        if self.normalize_derivative_features:
            X_selector = (X_selector-self.mini)/(self.maxi-self.mini)
        
        if include_unsup: unsup_loss = self.selector.loss(X_selector, y_selector)
        else: unsup_loss = None
        
        return fd_guidance, unsup_loss


# In[16]:


h_star = (u_star+1j*v_star)

fd_derivatives = np.hstack([h_star, h_star.real**2+h_star.imag**2, fd_h_x, fd_h_xx, fd_h_xxx])

semisup_model = SemiSupModel(
    network=ComplexNetwork(model=complex_model, index2features=feature_names, scale=True, lb=lb, ub=ub),
    selector=ComplexAttentionSelectorNetwork([len(feature_names), 50, 50, 1], prob_activation=F.softmax, bn=True),
    normalize_derivative_features=False,
    mini=torch.tensor(np.abs(fd_derivatives).min(axis=0), dtype=torch.cfloat),
    maxi=torch.tensor(np.abs(fd_derivatives).max(axis=0), dtype=torch.cfloat)
)

del h_star, fd_derivatives, fd_h_x, fd_h_xx, fd_h_xxx

semisup_model(X_train)


# In[17]:


selector=ComplexAttentionSelectorNetwork([len(feature_names), 50, 50, 1], prob_activation=F.softmax, bn=True)


# In[18]:


X_selector = (X_selector - semisup_model.mini) / (semisup_model.maxi-semisup_model.mini)
selector_optimizer = MADGRAD(selector.parameters(), lr=5e-2)
for i in range(50000):
    selector_optimizer.zero_grad()
    l = complex_mse(selector(X_selector), y_selector)
    l.backward(retain_graph=True)
    selector_optimizer.step()
    print(l.item())

torch.save(selector.state_dict(), './saved_path_inverse_nls/selector.pth')
