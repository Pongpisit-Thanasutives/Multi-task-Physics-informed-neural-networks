#!/usr/bin/env python
# coding: utf-8

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


# In[2]:


# Loading the KS sol
DATA_PATH = "../deephpms_data/KS_simple3.pkl"
data = pickle_load(DATA_PATH)
t = data['t']
x = data['x']
X, T = np.meshgrid(x, t)
Exact = data['u'].T

# Adding noise
noise_intensity = 0.01
noisy_xt = False

Exact = perturb(Exact, intensity=noise_intensity, noise_type="normal")
print("Perturbed Exact with intensity =", float(noise_intensity))

x_star = X.flatten()[:,None]
t_star = T.flatten()[:,None]
X_star = np.hstack((x_star, t_star))
u_star = Exact.T.flatten()[:,None]

if noisy_xt: 
    print("Noisy (x, t)")
    X_star = perturb(X_star, intensity=noise_intensity, noise_type="normal")
else: print("Clean (x, t)")

# Doman bounds
lb = X_star.min(axis=0)
ub = X_star.max(axis=0)

N = 5000
print(f"Fine-tuning with {N} samples")
idx = np.random.choice(X_star.shape[0], N, replace=False)
X_u_train = X_star[idx, :]
u_train = u_star[idx,:]

# Robust PCA
print("Running Robust PCA on X_u_train")
rpca = R_pca_numpy(X_u_train)
X_train_L, X_train_S = rpca.fit(tol=1e-20, max_iter=25000, iter_print=100, verbose=False)
print('Robust PCA Loss:', mean_squared_error(X_u_train, X_train_L+X_train_S))
X_train_S = to_tensor(X_train_S, False)

print("Running Robust PCA on X_u_train")
rpca = R_pca_numpy(u_train)
u_train_L, u_train_S = rpca.fit(tol=1e-20, max_iter=25000, iter_print=100, verbose=False)
print('Robust PCA Loss:', mean_squared_error(u_train, u_train_L+u_train_S))
u_train_S = to_tensor(u_train_S, False)

del rpca, X_train_L, u_train_L, X_star, u_star, Exact, data

# Convert to torch.tensor
X_u_train = to_tensor(X_u_train, True)
u_train = to_tensor(u_train, False)

# lb and ub are used in adversarial training
scaling_factor = 1.0
lb = scaling_factor*to_tensor(lb, False)
ub = scaling_factor*to_tensor(ub, False)

# Feature names, base on the symbolic regression results
feature_names=('uf', 'u_x', 'u_xx', 'u_xxxx'); feature2index = {}


# In[3]:


program = '''
-0.534833*u_xx-0.518928*u_xxxx-0.541081*uf*u_x
'''
pde_expr, variables = build_exp(program); print(pde_expr, variables)
mod = sympytorch.SymPyModule(expressions=[pde_expr]); mod.train()


# In[4]:


class RobustPINN(nn.Module):
    def __init__(self, model, loss_fn, index2features, scale=False, lb=None, ub=None, pretrained=False):
        super(RobustPINN, self).__init__()
        self.model = model
        if not pretrained: self.model.apply(self.xavier_init)
            
        # Robust Beta-PCA
        self.inp_rpca = RobustPCANN(beta=0.0, is_beta_trainable=True, inp_dims=2, hidden_dims=32)
        self.out_rpca = RobustPCANN(beta=0.0, is_beta_trainable=True, inp_dims=1, hidden_dims=32)
            
        self.callable_loss_fn = loss_fn
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
    
    def loss(self, X_input, X_input_S, y_input, y_input_S, update_network_params=True, update_pde_params=True):
        total_loss = []
        
        X_input = self.inp_rpca(X_input, X_input_S, normalize=True, is_clamp=False)
        y_input = self.out_rpca(y_input, y_input_S, normalize=True, is_clamp=False)
        
        grads_dict, u_t = self.grads_dict(X_input[:, 0:1], X_input[:, 1:2])
        # MSE Loss
        if update_network_params:
            mse_loss = F.mse_loss(grads_dict["uf"], y_input)
            total_loss.append(mse_loss)
        # PDE Loss
        if update_pde_params:
            l_eq = F.mse_loss(self.callable_loss_fn(**grads_dict).squeeze(-1), u_t)
            total_loss.append(l_eq)
            
        return total_loss
    
    def grads_dict(self, x, t):
        uf = self.forward(x, t)
        u_t = self.gradients(uf, t)[0]
        
        ### PDE Loss calculation ###
        derivatives = group_diff(uf, (x, t), self.diff_flag[1], function_notation="u", gd_init={"uf":uf})
        
        return derivatives, u_t
    
    def gradients(self, func, x):
        return grad(func, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones(func.shape))
    
    def neural_net_scale(self, inp): 
        return -1.0+2.0*(inp-self.lb)/(self.ub-self.lb)


# In[5]:


model = TorchMLP(dimensions=[2, 50, 50, 50 ,50, 50, 1], activation_function=nn.Tanh, bn=nn.LayerNorm, dropout=None)

# Pretrained model
semisup_model_state_dict = torch.load("./saved_path_inverse_small_KS/simple3_semisup_model_state_dict_250labeledsamples250unlabeledsamples.pth")
parameters = OrderedDict()
# Filter only the parts that I care about renaming (to be similar to what defined in TorchMLP).
inner_part = "network.model."
for p in semisup_model_state_dict:
    if inner_part in p:
        parameters[p.replace(inner_part, "")] = semisup_model_state_dict[p]
model.load_state_dict(parameters)

pinn = RobustPINN(model=model, loss_fn=mod, index2features=feature_names, scale=True, lb=lb, ub=ub, pretrained=True)

pinn = load_weights(pinn, "./saved_path_inverse_small_KS/noisy2_final_finetuned_doublebetarpca_pinn_5000.pth")
# pinn = load_weights(pinn, "tmp_f1.pth")

def closure():
    if torch.is_grad_enabled():
        optimizer2.zero_grad()
    losses = pinn.loss(X_u_train, X_train_S, u_train, u_train_S, update_network_params=True, update_pde_params=True)
    l = sum(losses)
    if l.requires_grad:
        l.backward(retain_graph=True)
    return l

def mtl_closure():
    n_obj = 2 # There are two tasks
    losses = pinn.loss(X_u_train, X_train_S, u_train, u_train_S, update_network_params=True, update_pde_params=True)
    updated_grads = []
    
    for i in range(n_obj):
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
        param.grad = (updated_grads[0][idx]+updated_grads[1][idx])
        
    return sum(losses)


# In[8]:


epochs1, epochs2 = 0, 50
# TODO: Save best state dict and training for more epochs.
optimizer1 = MADGRAD(pinn.parameters(), lr=1e-7, momentum=0.9)
pinn.train(); best_train_loss = 1e6

print('1st Phase optimization using Adam with PCGrad gradient modification')
for i in range(epochs1):
    optimizer1.step(mtl_closure)
    if (i % 10) == 0 or i == epochs1-1:
        l = mtl_closure()
        print("Epoch {}: ".format(i), l.item())
        
optimizer2 = torch.optim.LBFGS(pinn.parameters(), lr=1e-1, max_iter=500, max_eval=int(500*1.25), history_size=300, line_search_fn='strong_wolfe')
print('2nd Phase optimization using LBFGS')
for i in range(epochs2):
    optimizer2.step(closure)
    if (i % 10) == 0 or i == epochs2-1:
        save(pinn, "tmp.pth")
        l = closure()
        print("Epoch {}: ".format(i), l.item())


# In[9]:


pred_params = [x.item() for x in pinn.callable_loss_fn.parameters()]
print(pred_params)


# In[10]:


errs = 100*np.abs(np.array(pred_params)+1)
print(errs.mean(), errs.std())


# In[11]:


### Without AutoEncoder ###

# Clean Exact and (x, t)
# [-0.999634325504303, -0.9994997382164001, -0.9995566010475159]
# (0.04364450772603353, 0.005516461306387781)

# Pretrained with final_finetuned_pinn_5000 (not used)
# [-0.9996052980422974, -0.9995099902153015, -0.9995319247245789]
# 0.04509290059407552 0.0040754479422416435

# Noisy Exact and clean (x, t)
# [-0.9967969655990601, -0.9969800114631653, -0.9973703026771545]
# (0.2950906753540039, 0.023910676954986623)

# Noisy Exact and noisy (x, t)
# [-0.9975059032440186, -0.9962050914764404, -0.9969104528427124]
# 0.3126184145609538 0.05316856937153804


# In[12]:


### Without AutoEncoder & With Robust PCA ###

# Noisy Exact and clean (x, t)
# [-1.0003160238265991, -1.0005097389221191, -0.9997726678848267]
# (0.035103162129720054, 0.011791963483533422)

# Noisy Exact and noisy (x, t) + Robust PCA
# [-1.000351071357727, -0.9991073608398438, -0.9980993866920471]
# (0.08140603701273601 0.09209241474722876)


# In[ ]:




