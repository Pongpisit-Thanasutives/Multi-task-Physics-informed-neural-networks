#!/usr/bin/env python
# coding: utf-8
import matplotlib.pyplot as plt

# always import gbm_algos first !
import xgboost, lightgbm, catboost
from gplearn.genetic import SymbolicRegressor

# To access the contents of the parent dir
import sys; sys.path.insert(0, '../')
import os
from scipy.io import loadmat
from utils import *
from preprocess import *
from models import RobustPCANN

# Let's do facy optimizers
from optimizers import Lookahead, AdamGC, SGDGC
from madgrad import MADGRAD
from lbfgsnew import LBFGSNew

from pytorch_robust_pca import *

# Modify at /usr/local/lib/python3.9/site-packages/torch_lr_finder/lr_finder.py
from torch_lr_finder import LRFinder

# Tracking
from tqdm import trange

import sympy
import sympytorch

from pde_diff import *


# In[2]:


include_N_res = False

# DATA_PATH = '../deephpms_data/KS.mat'
DATA_PATH = '../PDE_FIND_experimental_datasets/kuramoto_sivishinky.mat'
X, T, Exact = space_time_grid(data_path=DATA_PATH, real_solution=True)
X_star, u_star = get_trainable_data(X, T, Exact)

# Doman bounds
lb = X_star.min(axis=0)
ub = X_star.max(axis=0)

N = 20000 # 20000, 30000, 60000
print(f"Fine-tuning with {N} samples")
# idx = np.random.choice(X_star.shape[0], N, replace=False)
idx = np.arange(N)
X_u_train = X_star[idx, :]
u_train = u_star[idx,:]

noise_intensity = 0.01
noisy_xt = False; noisy_labels = False
if noisy_xt: X_u_train = perturb(X_u_train, noise_intensity); print("Noisy (x, t)")
else: print("Clean (x, t)")
if noisy_labels: u_train = perturb(u_train, noise_intensity); print("Noisy labels")
else: print("Clean labels")

# Unsup data
if include_N_res:
    N_res = N//2
    idx_res = np.array(range(X_star.shape[0]-1))[~idx]
    idx_res = np.random.choice(idx_res.shape[0], N_res, replace=True)
    X_res = X_star[idx_res, :]
    print(f"Fine-tuning with {N_res} unsup samples")
    X_u_train = np.vstack([X_u_train, X_res])
    u_train = np.vstack([u_train, torch.rand(X_res.shape[0], 1) - 1000])
    # del X_res
else: print("Not including N_res")

# Convert to torch.tensor
X_u_train = to_tensor(X_u_train, True)
u_train = to_tensor(u_train, False)
X_star = to_tensor(X_star, True)
u_star = to_tensor(u_star, False)

# lb and ub are used in adversarial training
scaling_factor = 1.0
lb = scaling_factor*to_tensor(lb, False)
ub = scaling_factor*to_tensor(ub, False)

# Feature names, base on the symbolic regression results (only the important features)
feature_names=('uf', 'u_x', 'u_xx', 'u_xxxx'); feature2index = {}

# del X_star, u_star

# Noisy (x, t) and noisy labels
# PDE derived using STRidge to NN diff features
# u_t = (-0.912049 +0.000000i)u_xx
#     + (-0.909050 +0.000000i)u_xxxx
#     + (-0.951584 +0.000000i)uf*u_x

# Clean (x, t) but noisy labels
# PDE derived using STRidge to NN diff features
# u_t = (-0.942656 +0.000000i)u_xx
#     + (-0.900600 +0.000000i)u_xxxx
#     + (-0.919862 +0.000000i)uf*u_x

# Clean all
# PDE derived using STRidge to fd_derivatives
# u_t = (-0.995524 +0.000000i)uu_{x}
#     + (-1.006815 +0.000000i)u_{xx}
#     + (-1.005177 +0.000000i)u_{xxxx}

# program = '''
# -0.97*u_xx-0.902*u_xxxx-0.920*uf*u_x
# ''' 

program = '''
-1.006815*u_xx-1.005177*u_xxxx-0.995524*uf*u_x
'''

pde_expr, variables = build_exp(program); print(pde_expr, variables)
mod = sympytorch.SymPyModule(expressions=[pde_expr]); mod.train()

class RobustPINN(nn.Module):
    def __init__(self, model, loss_fn, index2features, scale=False, lb=None, ub=None, pretrained=False, noiseless_mode=True, init_cs=(0.5, 0.5), init_betas=(0.0, 0.0)):
        super(RobustPINN, self).__init__()
        self.model = model
        if not pretrained: self.model.apply(self.xavier_init)
        
        self.noiseless_mode = noiseless_mode
        if self.noiseless_mode: print("No denoising")
        else: print("With denoising method")
        
        self.in_fft_nn = None; self.out_fft_nn = None
        self.inp_rpca = None; self.out_rpca = None
        if not self.noiseless_mode:
            # FFTNN
            self.in_fft_nn = FFTTh(c=init_cs[0])
            self.out_fft_nn = FFTTh(c=init_cs[1])

            # Robust Beta-PCA
            self.inp_rpca = RobustPCANN(beta=init_betas[0], is_beta_trainable=True, inp_dims=2, hidden_dims=32)
            self.out_rpca = RobustPCANN(beta=init_betas[1], is_beta_trainable=True, inp_dims=1, hidden_dims=32)
        
        self.callable_loss_fn = loss_fn
        self.init_parameters = [nn.Parameter(torch.tensor(x.item())) for x in loss_fn.parameters()]
        print("Initial parameters", self.init_parameters)
        self.param0 = self.init_parameters[0]
        self.param1 = self.init_parameters[1]
        self.param2 = self.init_parameters[2]
        del self.callable_loss_fn, self.init_parameters
        
        self.index2features = index2features; self.feature2index = {}
        for idx, fn in enumerate(self.index2features): self.feature2index[fn] = str(idx)
        self.scale = scale; self.lb, self.ub = lb, ub
        self.diff_flag = diff_flag(self.index2features)
        
    def xavier_init(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        
    def forward(self, x, t):
        H = torch.cat([x, t], dim=1)
        if self.scale: H = self.neural_net_scale(H)
        return self.model(H)
    
    def loss(self, X_input, X_input_noise, y_input, y_input_noise, update_network_params=True, update_pde_params=True):
        # Denoising process
        if not self.noiseless_mode:
            # (1) Denoising FFT on (x, t)
            # This line returns the approx. recon.
            X_input_noise = cat(torch.fft.ifft(self.in_fft_nn(X_input_noise[1])*X_input_noise[0]).real.reshape(-1, 1), 
                                torch.fft.ifft(self.in_fft_nn(X_input_noise[3])*X_input_noise[2]).real.reshape(-1, 1))
            X_input_noise = X_input-X_input_noise
            X_input = self.inp_rpca(X_input, X_input_noise, normalize=True)
            
            # (2) Denoising FFT on y_input
            y_input_noise = y_input-torch.fft.ifft(self.out_fft_nn(y_input_noise[1])*y_input_noise[0]).real.reshape(-1, 1)
            y_input = self.out_rpca(y_input, y_input_noise, normalize=True)
        
        grads_dict, u_t = self.grads_dict(X_input[:, 0:1], X_input[:, 1:2])
        
        total_loss = []
        # MSE Loss
        if update_network_params:
            mse_loss = F.mse_loss(grads_dict["uf"], y_input)
            total_loss.append(mse_loss)
            
        # PDE Loss
        if update_pde_params:
            u_t_pred = (self.param2*grads_dict["uf"]*grads_dict["u_x"])+(self.param1*grads_dict["u_xx"])+(self.param0*grads_dict["u_xxxx"])
            l_eq = F.mse_loss(u_t_pred, u_t)
            total_loss.append(l_eq)
            
        return total_loss
    
    def grads_dict(self, x, t):
        uf = self.forward(x, t)
        u_t = self.gradients(uf, t)[0]
        u_x = self.gradients(uf, x)[0]
        u_xx = self.gradients(u_x, x)[0]
        u_xxx = self.gradients(u_xx, x)[0]
        u_xxxx = self.gradients(u_xxx, x)[0]        
        return {"uf":uf, "u_x":u_x, "u_xx":u_xx, "u_xxxx":u_xxxx}, u_t
    
    def get_selector_data(self, x, t):
        uf = self.forward(x, t)
        u_t = self.gradients(uf, t)[0]
        
        ### PDE Loss calculation ###
        # 'uf', 'u_x', 'u_xx', 'u_xxxx', 'u_xxx'
        u_x = self.gradients(uf, x)[0]
        u_xx = self.gradients(u_x, x)[0]
        u_xxx = self.gradients(u_xx, x)[0]
        u_xxxx = self.gradients(u_xxx, x)[0]
        u_xxxxx = self.gradients(u_xxxx, x)[0]
        derivatives = []
        derivatives.append(uf)
        derivatives.append(u_x)
        derivatives.append(u_xx)
        derivatives.append(u_xxx)
        derivatives.append(u_xxxx)
        derivatives.append(u_xxxxx)
        
        return torch.cat(derivatives, dim=1), u_t
    
    def gradients(self, func, x):
        return grad(func, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones(func.shape))
    
    def neural_net_scale(self, inp): 
        return -1.0+2.0*(inp-self.lb)/(self.ub-self.lb)


# In[5]:


noiseless_mode = False
model = TorchMLP(dimensions=[2, 50, 50, 50 ,50, 50, 1], activation_function=nn.Tanh, bn=nn.LayerNorm, dropout=None)

# Pretrained model
semisup_model_state_dict = cpu_load("../inverse_KS/saved_path_inverse_ks/semisup_model_with_LayerNormDropout_without_physical_reg_trained30000labeledsamples_trained0unlabeledsamples.pth")
parameters = OrderedDict()
# Filter only the parts that I care about renaming (to be similar to what defined in TorchMLP).
inner_part = "network.model."
for p in semisup_model_state_dict:
    if inner_part in p:
        parameters[p.replace(inner_part, "")] = semisup_model_state_dict[p]
model.load_state_dict(parameters)

pinn = RobustPINN(model=model, loss_fn=mod, 
                  index2features=feature_names, scale=True, lb=lb, ub=ub, 
                  pretrained=True, noiseless_mode=noiseless_mode)
pinn = load_weights(pinn, "./new_saved_path/KS_rudy_pretrained_pinn_2ndrun.pth")

# assigning the prefered loss_fn
model = pinn.model
pinn = RobustPINN(model=model, loss_fn=mod, 
                  index2features=feature_names, scale=True, lb=lb, ub=ub, 
                  pretrained=True, noiseless_mode=noiseless_mode)


# #### Use STRidge to discover the hidden relation (on top of the pretrained solver net)

lets_pretrain = False

if lets_pretrain:
    xx, tt = X_u_train[:, 0:1], X_u_train[:, 1:2]

    pretraining_optimizer = LBFGSNew(pinn.model.parameters(),
                                     lr=1e-1, max_iter=500,
                                     max_eval=int(500*1.25), history_size=300,
                                     line_search_fn=True, batch_mode=False)

    model.train()
    for i in range(5): # 1, 5, 200
        def pretraining_closure():
            global xx, tt, u_train
            if torch.is_grad_enabled(): pretraining_optimizer.zero_grad()
            mse_loss = F.mse_loss(pinn(xx, tt), u_train)
            if mse_loss.requires_grad: mse_loss.backward(retain_graph=False)
            return mse_loss

        pretraining_optimizer.step(pretraining_closure)

        if (i%1)==0:
            l = pretraining_closure()
            curr_loss = l.item()
            print("Epoch {}: ".format(i), curr_loss)

# xx, tt = X_u_train[:, 0:1], X_u_train[:, 1:2]

NUMBER = 21000 # 20000, 21000
NUMBER = min(NUMBER, X_star.shape[0])
xx, tt = X_star[:NUMBER, 0:1], X_star[:NUMBER, 1:2]
# xx, tt = torch.FloatTensor(xx).requires_grad_(True), torch.FloatTensor(tt).requires_grad_(True)

uf = pinn(xx, tt)
u_t = pinn.gradients(uf, tt)[0]

### PDE Loss calculation ###
# 'uf', 'u_x', 'u_xx', 'u_xxxx', 'u_xxx'
u_x = pinn.gradients(uf, xx)[0]
u_xx = pinn.gradients(u_x, xx)[0]
u_xxx = pinn.gradients(u_xx, xx)[0]
u_xxxx = pinn.gradients(u_xxx, xx)[0]
# u_xxxxx = pinn.gradients(u_xxxx, xx)[0]

derivatives = []
derivatives.append(uf)
derivatives.append(u_x)
derivatives.append(u_xx)
# derivatives.append(u_xxx)
derivatives.append(u_xxxx)
# derivatives.append(u_xxxxx)
derivatives = torch.cat(derivatives, dim=1)

derivatives = derivatives.detach().numpy()
u_t = u_t.detach().numpy()


# In[10]:


feature_names = ["uf", "u_x", "u_xx", "u_xxxx"]

X_input = derivatives
y_input = u_t

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_input = poly.fit_transform(X_input)

poly_feature_names = poly.get_feature_names(feature_names)
for i, f in enumerate(poly_feature_names): poly_feature_names[i] = f.replace(" ", "*")

# # Set normalize=1
# TrainSTRidge(X_input[:, :], y_input, 1e-6, 100, normalize=1) 1st run
# TrainSTRidge(X_input[:, :], y_input, 1e-6, 10, normalize=1) 2nd run
w = TrainSTRidge(X_input[:, :], y_input, 1e-6, 10, normalize=1)
print("PDE derived using STRidge")
print_pde(w, poly_feature_names[:])

_, x_fft, x_PSD = fft1d_denoise(X_u_train[:, 0:1], c=-5, return_real=True)
_, t_fft, t_PSD = fft1d_denoise(X_u_train[:, 1:2], c=-5, return_real=True)
_, u_train_fft, u_train_PSD = fft1d_denoise(u_train, c=-5, return_real=True)

x_fft, x_PSD = x_fft.detach(), x_PSD.detach()
t_fft, t_PSD = t_fft.detach(), t_PSD.detach()


# In[15]:


def closure():
    if torch.is_grad_enabled():
        optimizer2.zero_grad()
    losses = pinn.loss(X_u_train, (x_fft, x_PSD, t_fft, t_PSD), u_train, (u_train_fft, u_train_PSD), update_network_params=True, update_pde_params=True)
    l = sum(losses)
    if l.requires_grad:
        l.backward(retain_graph=True)
    return l

def mtl_closure():
    losses = pinn.loss(X_u_train, (x_fft, x_PSD, t_fft, t_PSD), u_train, (u_train_fft, u_train_PSD), update_network_params=True, update_pde_params=True)
    updated_grads = []
    
    for i in range(len(losses)):
        optimizer1.zero_grad()
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
        param.grad = updated_grads[0][idx]+updated_grads[1][idx]
        
    return sum(losses)


# In[16]:


# pinn.loss(X_u_train, (x_fft, x_PSD, t_fft, t_PSD), u_train, (u_train_fft, u_train_PSD), update_network_params=True, update_pde_params=True)


# In[17]:


epochs1, epochs2 = 200, 20
# TODO: Save best state dict and training for more epochs.
optimizer1 = MADGRAD(pinn.parameters(), lr=1e-5, momentum=0.9)
pinn.train(); best_train_loss = 1e6

print('1st Phase optimization using Adam with PCGrad gradient modification')
for i in range(epochs1):
    optimizer1.step(mtl_closure)
    if (i % 10) == 0 or i == epochs1-1:
        l = mtl_closure()
        print("Epoch {}: ".format(i), l.item())
        print(pinn.param0, pinn.param1, pinn.param2)
        
optimizer2 = torch.optim.LBFGS(pinn.parameters(), lr=1e-1, max_iter=500, max_eval=int(500*1.25), history_size=300, line_search_fn='strong_wolfe')
print('2nd Phase optimization using LBFGS')
for i in range(epochs2):
    optimizer2.step(closure)
    if (i % 5) == 0 or i == epochs2-1:
        l = closure()
        print("Epoch {}: ".format(i), l.item())

pred_params = [pinn.param0.item(), pinn.param1.item(), pinn.param2.item()]
print(pred_params)

errs = 100*np.abs(np.array(pred_params)+1)
print(errs.mean(), errs.std())

save(pinn, "./new_saved_path/KS_rudy_final_finetuned_dft_pinn.pth")
