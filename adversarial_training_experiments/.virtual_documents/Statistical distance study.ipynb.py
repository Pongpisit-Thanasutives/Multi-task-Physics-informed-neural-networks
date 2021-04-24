import sys; sys.path.insert(1, '../')
from utils import *
from pytorch_stats_loss import torch_wasserstein_loss, torch_energy_loss

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.datasets import make_circles
from scipy.stats import *

# for ploting only
get_ipython().run_line_magic("matplotlib", " inline")
import matplotlib.pyplot as plt
import cufflinks as cf
import plotly.express as px
from plotly.offline import download_plotlyjs, init_notebook_mode
import plotly.graph_objects as go
init_notebook_mode(connected=True)
cf.go_offline()

# tracking
from tqdm import trange


def get_label(Xs): return multivariate_normal.pdf(Xs, mean=np.array([1.0, 0]), cov=np.eye(X.shape[1]))


N = 2000
X = sampling_unit_circle(100)
y = get_label(X)
plt.scatter(X[:, 0], X[:, 1])
plt.show()


# fig = go.Figure(data=go.Contour(z=y, x=X[:, 0], y=X[:, 1]))
# fig.show()


# converting data to torch.tensor 
X = to_tensor(X, False)
y = to_tensor(y, False).reshape(-1, 1)

# setting the networks
solver = TorchMLP([2, 32, 32, 1])
generator = TorchMLP([2, 32, 32, 2])
solver_optimizer = torch.optim.Adam(solver.parameters(), lr=5e-3)
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=3e-4)

# training configs
epochs = 10000
lb, ub = -2, 2
adv_f = 100


for i in trange(0, epochs):
    if iget_ipython().run_line_magic("adv_f==0:", "")
        # train generator
        print("Training the generator")
        for _ in range(2000):
            solver.eval()
            generator.train()
            solver_optimizer.zero_grad()
            generator_optimizer.zero_grad()
            X_gen = generator(X)
            X_gen = scale_to_range(X_gen, lb, ub)
            y_gen = to_tensor(get_label(to_numpy(X_gen)), False).reshape(-1, 1)
            distance_loss = torch_energy_loss(X_gen[:, 0], X[:, 0])+torch_energy_loss(X_gen[:, 1], X[:, 1])
            pred = solver(X_gen)
            adv_loss = -F.mse_loss(pred, y_gen)
            generator_loss = distance_loss+adv_loss
            generator_loss.backward(retain_graph=True)
            generator_optimizer.step()

    # train solver
    solver.train()
    generator.eval()
    solver_optimizer.zero_grad()
    generator_optimizer.zero_grad()
    X_train = torch.cat([X, X_gen], dim=0).detach()
    y_train = torch.cat([y, y_gen], dim=0).detach()
    pred = solver(X_train)
    solver_loss = F.mse_loss(pred, y_train)
    solver_loss.backward(retain_graph=True)
    solver_optimizer.step()
    
    if iget_ipython().run_line_magic("1000==0:", "")
        print("Epoch:", str(i), solver_loss.item())
        print("Epoch:", str(i), generator_loss.item())


plt.scatter(to_numpy(X)[:, 0], to_numpy(X)[:, 1])
plt.scatter(to_numpy(X_gen)[:, 0], to_numpy(X_gen)[:, 1])
plt.show()



