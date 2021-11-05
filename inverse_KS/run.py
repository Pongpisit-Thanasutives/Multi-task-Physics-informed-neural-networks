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

# Let's do facy optimizers
from optimizers import Lookahead, AdamGC, SGDGC
from madgrad import MADGRAD
from lbfgsnew import LBFGSNew

# Modify at /usr/local/lib/python3.9/site-packages/torch_lr_finder/lr_finder.py
from torch_lr_finder import LRFinder

# Tracking
from tqdm import trange

import sympy
import sympytorch


# In[2]:


include_N_res = False

# DATA_PATH = '../deephpms_data/KS.mat'
DATA_PATH = '../PDE_FIND_experimental_datasets/kuramoto_sivishinky.mat'
X, T, Exact = space_time_grid(data_path=DATA_PATH, real_solution=True)
X_star, u_star = get_trainable_data(X, T, Exact)

# Doman bounds
lb = X_star.min(axis=0)
ub = X_star.max(axis=0)

N = 30000 #30000
print(f"Fine-tuning with {N} samples")
# idx = np.random.choice(X_star.shape[0], N, replace=False)
idx = np.arange(N)
X_u_train = X_star[idx, :]
u_train = u_star[idx,:]

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

del X_star, u_star


# In[3]:


# Type the equation got from the symbolic regression step
# No need to save the eq save a pickle file before
program = "-0.90*X0*X1-0.91*X2-0.89*X3"
pde_expr, variables,  = build_exp(program); print(pde_expr, variables)
mod = sympytorch.SymPyModule(expressions=[pde_expr]); mod.train()


# In[4]:


class PINN(nn.Module):
    def __init__(self, model, loss_fn, index2features, scale=False, lb=None, ub=None, pretrained=False):
        super(PINN, self).__init__()
        self.model = model
        if not pretrained: self.model.apply(self.xavier_init)
        self.callable_loss_fn = loss_fn
        self.init_parameters = [x.item() for x in self.callable_loss_fn.parameters()]
        self.params0 = nn.Parameter(torch.tensor(self.init_parameters[0]))
        self.params1 = nn.Parameter(torch.tensor(self.init_parameters[1]))
        self.params2 = nn.Parameter(torch.tensor(self.init_parameters[2]))
        del self.callable_loss_fn
        
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
    
    def loss(self, x, t, y_input, update_network_params=True, update_pde_params=True):
        total_loss = []
        grads_dict, u_t = self.grads_dict(x, t)
        # MSE Loss
        if update_network_params:
            mse_loss = F.mse_loss(grads_dict['X'+self.feature2index['uf']], y_input)
            total_loss.append(mse_loss)
        # PDE Loss
        if update_pde_params:
            l_eq = u_t-(self.params0*grads_dict['X0']*grads_dict['X1']+self.params1*grads_dict['X2']+self.params2*grads_dict['X3'])
            l_eq = (l_eq**2).mean()
            total_loss.append(l_eq)
            
        return total_loss
    
    def grads_dict(self, x, t):
        uf = self.forward(x, t)
        u_t = self.gradients(uf, t)[0]
        u_x = self.gradients(uf, x)[0]
        u_xx = self.gradients(u_x, x)[0]
        u_xxx = self.gradients(u_xx, x)[0]
        u_xxxx = self.gradients(u_xxx, x)[0]
        
        ### PDE Loss calculation ###
        # Without calling grad
        derivatives = {'X0': uf, 'X1': u_x, 'X2': u_xx, 'X3': u_xxxx}
        
        return derivatives, u_t
    
    def gradients(self, func, x):
        return grad(func, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones(func.shape))

    def neural_net_scale(self, inp): 
        return 2*(inp-self.lb)/(self.ub-self.lb)-1


# In[5]:


model = TorchMLP(dimensions=[2, 50, 50, 50 ,50, 50, 1], activation_function=nn.Tanh, bn=nn.LayerNorm, dropout=None)

# Pretrained model
semisup_model_state_dict = cpu_load("./saved_path_inverse_ks/semisup_model_with_LayerNormDropout_without_physical_reg_trained30000labeledsamples_trained0unlabeledsamples.pth")
parameters = OrderedDict()
# Filter only the parts that I care about renaming (to be similar to what defined in TorchMLP).
inner_part = "network.model."
for p in semisup_model_state_dict:
    if inner_part in p:
        parameters[p.replace(inner_part, "")] = semisup_model_state_dict[p]
model.load_state_dict(parameters)

pinn = PINN(model=model, loss_fn=mod, index2features=feature_names, scale=True, lb=lb, ub=ub, pretrained=True)
pinn.load_state_dict(torch.load("./saved_path_inverse_ks/final_finetuned_pinn.pth"), strict=False)

def closure():
    if torch.is_grad_enabled():
        optimizer2.zero_grad()
    losses = pinn.loss(X_u_train[:, 0:1], X_u_train[:, 1:2], u_train, update_network_params=True, update_pde_params=True)
    l = sum(losses)
    if l.requires_grad:
        l.backward(retain_graph=True)
    return l

def mtl_closure():
    n_obj = 2 # There are two tasks
    losses = pinn.loss(X_u_train[:, 0:1], X_u_train[:, 1:2], u_train, update_network_params=True, update_pde_params=True)
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


epochs1, epochs2 = 200, 30
# TODO: Save best state dict and training for more epochs.
optimizer1 = MADGRAD(pinn.parameters(), lr=1e-6, momentum=0.9)
pinn.train(); best_train_loss = 1e6

print('1st Phase optimization using Adam with PCGrad gradient modification')
for i in range(epochs1):
    optimizer1.step(mtl_closure)
    l = mtl_closure()
    if (i % 10) == 0 or i == epochs1-1:
        print("Epoch {}: ".format(i), l.item())
        print(pinn.params0, pinn.params1, pinn.params2)


# In[ ]:


optimizer2 = torch.optim.LBFGS(pinn.parameters(), lr=1e-1, max_iter=300, max_eval=int(300*1.25), history_size=150, line_search_fn='strong_wolfe')
print('2nd Phase optimization using LBFGS')
for i in range(epochs2):
    optimizer2.step(closure)
    l = closure()
    if (i % 5) == 0 or i == epochs2-1:
        print("Epoch {}: ".format(i), l.item())
print(pinn.params0, pinn.params1, pinn.params2)


# In[ ]:


pred_params = [pinn.params0.item(), pinn.params1.item(), pinn.params2.item()]
print(pred_params)


# In[ ]:


results = np.array([(abs(pred_params[0]+1))*100, (abs(pred_params[1]+1))*100, (abs(pred_params[2]+1))*100])
results.mean(), results.std()

# torch.save(pinn.state_dict(), "./saved_path_inverse_ks/final_finetuned_pinn.pth")
