import os; os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pickle
from glob import glob as flist
from collections import Counter
from sympy import *
from sympy.parsing.sympy_parser import parse_expr
from sympy.core import evaluate

### Model-related imports ###
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader, Dataset
from collections import OrderedDict

import math
import numpy as np
import pandas as pd
from sklearn.metrics import *
from sklearn.preprocessing import PolynomialFeatures
from pyGRNN import feature_selection as FS

import pcgrad
from pytorch_stats_loss import torch_wasserstein_loss, torch_energy_loss 

# Finite difference method
from findiff import FinDiff, coefficients, Coefficient

def search_files(directory='.', extension=''):
    extension = extension.lower()
    for dirpath, dirnames, files in os.walk(directory):
        for name in files:
            if extension and name.lower().endswith(extension):
                print(os.path.join(dirpath, name))
            elif not extension:
                print(os.path.join(dirpath, name))

## Saving ###
def pickle_save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print('Saved to', str(path))

### Loading ###
def pickle_load(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    print('Loaded from', str(path))
    return obj

def build_exp(program):
    print("Please ensure that all coefs are different")
    x = Symbol("x"); y = Symbol("y")
    
    local_dict = {
        "add": Add,
        "mul": Mul,
        "sub": Lambda((x, y), x - y),
        "div": Lambda((x, y), x/y),
    }
    
    exp = simplify(parse_expr(str(program), local_dict=local_dict))
    exp = exp.subs(Integer(-1), Float(-1.0, precision=53))
    exp = exp.subs(Integer(+1), Float(1.0, precision=53))
    variables = exp.atoms(Symbol)
    
    return exp, variables

def manipulate_expr(expr):
    for coeff in expr.atoms(Number):
        if coeff < 0.005 and coeff > 0:
            new_coeff = log(coeff)
            with evaluate(False):
                new_coeff = exp(new_coeff)
                expr = expr.subs(coeff, new_coeff)
    with evaluate(False):
        return expr

def inverse_dict(dic):
    return {v: k for k, v in dic.items()}

def string2int(s):
    out = 0
    for i in range(len(s)):
        out += ord(s[i])
    return out

def scientific2string(x):
    return format(x, '.1e')

def convert_listoftuples_dict(tup, di={}):
    for a, b in tup: di[a]=b
    return di

def dimension_slicing(a_tensor):
    c = a_tensor.shape[-1]
    out = []
    for i in range(1, c+1): out.append(a_tensor[:, i-1:i])
    return out

def cat(*args):
    return torch.cat(args, dim=-1)

def get_feature(a_tensor, dim):
    return a_tensor[:, dim:dim+1]

def cpu_load(a_path):
    return torch.load(a_path, map_location="cpu")

def gpu_load(a_path):
    return torch.load(a_path, map_location="cuda")

def is_nan(a_tensor):
    return torch.isnan(a_tensor).any().item()

def to_column_vector(arr):
    return arr.flatten()[:, None]

def to_tensor(arr, g=True):
    return torch.tensor(arr).float().requires_grad_(g)

def to_complex_tensor(arr, g=True):
    return torch.tensor(arr, dtype=torch.cfloat).requires_grad_(g)

def to_numpy(a_tensor):
    return a_tensor.detach().numpy()

def sampling_unit_circle(N):
    points = []
    for _ in range(N):
        length = np.sqrt(np.random.uniform(0, 1)); angle = np.pi * np.random.uniform(0, 2)
        points.append([length * np.cos(angle), length * np.sin(angle)])
    return np.array(points)

def sampling_from_rows(a_tensor, N):
    r = a_tensor.shape[0]
    idxs = np.random.choice(r, N, replace=False) 
    return a_tensor[idxs, :]

# is the mini and maxi is going to be dynamics during the training but not trainable because of there is no grad.
def minmax_normalize(features):
    mini = torch.min(features, axis=0)[0]
    maxi = torch.max(features, axis=0)[0]
    features_std = (features-mini) / (maxi-mini) 
    return features_std

def scale_to_range(features, lb, ub):
    scaled_features = minmax_normalize(features)
    scaled_features = (ub-lb)*scaled_features + lb  
    return scaled_features

# is the mini and maxi is going to be dynamics during the training but not trainable because of there is no grad.
def numpy_minmax_normalize(arr):
    mini = np.min(arr, axis=0)
    maxi = np.max(arr, axis=0)
    return (arr-mini)/(maxi-mini)

def numpy_scale_to_range(arr, lb, ub):
    scaled_arr = numpy_minmax_normalize(arr)
    scaled_arr = (ub-lb)*scaled_arr + lb
    return scaled_arr

def cap_values(a_tensor, lb, ub):
    return (a_tensor-lb)/(ub-lb)

def diff_order(dterm):
    return dterm.split("_")[-1][::-1]

def diff_flag(index2feature):
    dd = {0:[], 1:[]}
    for t in index2feature:
        if '_' not in t: dd[0].append(t)
        else: dd[1].append(diff_order(t))
    return dd

def diff(func, inp):
    return grad(func, inp, create_graph=True, retain_graph=True, grad_outputs=torch.ones(func.shape, dtype=func.dtype))[0]

# Careful that there is no delta[i] = 0
# Use this function to approximate a higher-order derivative
def fd_diff(func, xx, dim=0):
    return torch.diff(func, dim=dim, n=1)/(torch.diff(xx, dim=dim))

def complex_diff(func, inp, return_complex=True):
    if return_complex: return diff(func.real, inp)+1j*diff(func.imag, inp)
    else: return cat(diff(func.real, inp), diff(func.imag, inp))

def finite_diff(func, axis, delta, diff_order=1, acc_order=2):
    assert axis in range(len(func.shape))
    return FinDiff(axis, delta, diff_order, acc=acc_order)(func)

class FinDiffCalculator:
    def __init__(self, X, T, Exact, dx=None, dt=None, acc_order=2):
        print("Do not use this class with complex-valued input arrays.")
        self.X = X; self.T = T; self.Exact = Exact.T
        if dx is not None: self.dx = dx
        else: self.dx = self.X[0, :][1]-self.X[0, :][0]
        print('dx =', self.dx)
        if dt is not None: self.dt = dt
        else: self.dt = self.T[:, 0][1]-self.T[:, 0][0]
        print('dt =', self.dt)
        self.deltas = [self.dx, self.dt]
        self.acc_order = acc_order
    # Cal d_dt using this function
    def finite_diff(self, axis, diff_order=1):
        return to_column_vector(FinDiff(axis, self.deltas[axis], diff_order, acc=self.acc_order)(self.Exact).T)
    def finite_diff_from_feature_names(self, index2feature):
        out = {}
        for f in index2feature:
            if '_' not in f:
                if f == 'uf' or f == 'hf': out[f] = to_column_vector(self.Exact.T)
                elif f == 'x': out[f] = to_column_vector(self.X)
                elif f == '|uf|' or f == '|hf|': out[f] = to_column_vector((self.Exact.real**2+self.Exact.imag**2).T)
                else: raise NotImplementedError
            else:
                counter = Counter(f.split('_')[1])
                if len(counter.keys())==1 and 'x' in counter.keys():
                    out[f] = (self.finite_diff(axis=0, diff_order=counter['x']))
                else: raise NotImplementedError
        return out

def train_val_split(a_tensor, train_ratio=0.8):
    train_len = int(0.8*a_tensor.shape[0])
    val_len = a_tensor.shape[0]-train_len
    train_idx, val_idx = torch.utils.data.random_split(np.arange(a_tensor.shape[0]), lengths=[train_len, val_len])
    train_idx = torch.tensor(train_idx)
    val_idx = torch.tensor(val_idx)
    return a_tensor[train_idx], a_tensor[val_idx]

# this function is supposed to be used with the TrainingDataset class
def get_dataloader(X_train, y_train, bs, N_sup=2000):
    return DataLoader(TrainingDataset(X_train, y_train, N_sup=N_sup), batch_size=bs)

# simple dataset class containing pair (x, y)
class XYDataset(Dataset):
    def __init__(self, X_data, y_data):
        super(XYDataset, self).__init__()
        assert X_data.shape[0] == y_data.shape[0]
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, idx):
        return self.X_data[idx, :], self.y_data[idx, :]

    def __len__(self,):
        return self.X_data.shape[0]

class TrainingDataset(Dataset):
    def __init__(self, X_train, y_train, N_sup=2000):
        super(TrainingDataset, self).__init__()
        self.X = X_train
        self.y = y_train
        self.N_sup=N_sup
        
    def __getitem__(self, index):
        if index > self.N_sup-1:
            return self.X[index], self.y[index]
        return self.X[index], self.y[index]

    def __len__(self):
        return self.X.shape[0]
    
class LadderLoss(nn.Module):
    def __init__(self, return_list=False):
        super().__init__()
        self.return_list = return_list
        
    def forward(self, outputs, labels):
        valid_index = torch.where(~(labels<-999))[0]
        tmp_out = outputs[0][valid_index]
        tmp_labels = labels[valid_index]
        unsup_loss = outputs[1]
        mse_loss = 0.0
        if tmp_out.shape[0] > 0 and tmp_labels.shape[0] > 0:
            mse_loss = F.mse_loss(tmp_out, tmp_labels)
        if not self.return_list: return mse_loss+unsup 
        return [mse_loss, unsup_loss] 

class LadderUncertLoss(nn.Module):
    def __init__(self, n_task):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros((n_task)))
 
    def forward(self, outputs, labels):
        weights = torch.exp(self.log_vars)
        mse_loss = F.mse_loss(outputs[0], labels).unsqueeze(0)
        unsup_loss = outputs[1].unsqueeze(0)
        losses = torch.cat([mse_loss, unsup_loss])
        return weights.dot(losses)

def distance_loss(inputs, targets, distance_function=torch_wasserstein_loss):
     total_loss = 0.0
     assert inputs.shape == targets.shape
     for i in range(inputs.shape[1]):
         total_loss += distance_function(inputs[:, i], targets[:, i])
     return total_loss

### Model-related code base ###
class CrossStich(nn.Module):
    def __init__(self,):
        super(CrossStich, self).__init__()
        self.transform = nn.Parameter(data=torch.eye(2), requires_grad=True)
    def forward(self, input_1, input_2):
        return self.transform[0][0]*input_1 + self.transform[0][1]*input_2, self.transform[1][0]*input_1 + self.transform[1][1]*input_2
    
def sparse_layer(in_dim, out_dim, sparsity):
    return SparseWeights(nn.Linear(in_dim, out_dim), sparsity=sparsity)

# This should be the activation function for the selector network. The module outputs a feature masking tensor.
class ThresholdSoftmax(nn.Module):
    def __init__(self, th=0.1):
        super(ThresholdSoftmax, self).__init__()
        self.sm = nn.Softmax(dim=-1)
        self.th = th
        self.prob = None
    def forward(self, inn):
        self.prob = self.sm(inn).mean(dim=0)
        thres = self.prob[torch.argsort(self.prob, descending=True)[3]]
        self.prob = torch.where(self.prob > thres, self.prob, torch.tensor([self.th]).float()) 
        samples = torch.sort(torch.multinomial(self.prob, 3))[0]
        return samples

class Swish(nn.Module):
    def __init__(self,):
        super(Swish, self).__init__()
    
    def forward(self, x):
        return x*torch.sigmoid(x)

class AconC(nn.Module):
    r""" ACON activation (activate or not).
    # AconC: (p1*x-p2*x) * sigmoid(beta*(p1*x-p2*x)) + p2*x, beta is a learnable parameter
    # according to "Activate or Not: Learning Customized Activation" <https://arxiv.org/pdf/2009.04759.pdf>.
    """

    def __init__(self, width):
        super().__init__()
        self.p1 = nn.Parameter(torch.randn(1, width, 1, 1))
        self.p2 = nn.Parameter(torch.randn(1, width, 1, 1))
        self.beta = nn.Parameter(torch.ones(1, width, 1, 1))

    def forward(self, x):
        return (self.p1 * x - self.p2 * x) * torch.sigmoid(self.beta * (self.p1 * x - self.p2 * x)) + self.p2 * x

def simple_solver_model(hidden_nodes):
    model = nn.Sequential(nn.Linear(2, hidden_nodes),
                        nn.Tanh(),
                        nn.Linear(hidden_nodes, hidden_nodes),
                        nn.Tanh(),
                        nn.Linear(hidden_nodes, hidden_nodes),
                        nn.Tanh(),
                        nn.Linear(hidden_nodes, hidden_nodes),
                        nn.Tanh(),
                        nn.Linear(hidden_nodes, 1))
    return model 

def evaluate_network_mse(network, X_star, u_star):
    return ((network(X_star[:, 0:1], X_star[:, 1:2]).detach() - u_star)**2).mean().item()

def evaluate_ladder_network_mse(network, X_star, u_star):
    return ((network(X_star[:, 0:1], X_star[:, 1:2])[0].detach() - u_star)**2).mean().item()

class TorchMLP(nn.Module):
    def __init__(self, dimensions, bias=True,activation_function=nn.Tanh, bn=None, dropout=None, inp_drop=False, final_activation=None):
        super(TorchMLP, self).__init__()
        print("Using old implementation of TorchMLP. See models.py for more new model-related source code.")
        self.model  = nn.ModuleList()
        # Can I also use the LayerNorm with elementwise_affine=True
        # This should be a callable module.
        self.activation_function = activation_function()
        self.bn = bn
        if dropout is not None and inp_drop: self.inp_dropout = dropout
        else: self.inp_dropout = None
        for i in range(len(dimensions)-1):
            self.model.append(nn.Linear(dimensions[i], dimensions[i+1], bias=bias))
            if self.bn is not None and i!=len(dimensions)-2:
                self.model.append(self.bn(dimensions[i+1]))
                if dropout is not None:
                    self.model.append(dropout)
            if i==len(dimensions)-2: break
            self.model.append(activation_function())
        if final_activation is not None:
            self.model.append(final_activation())
        self.model.apply(self.xavier_init)

    def xavier_init(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        if hasattr(self, 'inp_dropout'):
            if self.inp_dropout is not None:
                x = self.inp_dropout(x)
        for i, l in enumerate(self.model): 
            x = l(x)
        return x

class SklearnModel:
    def __init__(self, model, X_train=None, y_train=None, feature_names=None):
        self.model = model
        self.feature_names = feature_names
        self.is_train = False
        self.X_train_shape = None 
        if X_train is not None and y_train is not None:
            self.train(X_train, y_train)
        self.feature_importances = None
    def train(self, X_np, y_np):
        if self.is_train:
            print("This model was already trained")
        else:
            self.model.fit(X_np, y_np)
            self.X_train_shape = X_np.shape
            print("Done training")
            print("Training MSE:", mean_squared_error(self.model.predict(X_np), y_np))
    def feature_importance(self, feature_names=None):
        if feature_names is not None:
            self.feature_names = feature_names
        if self.feature_names is None:
            self.feature_names = [str(i) for i in range((self.X_train_shape[1]))]
        ranking = np.argsort(self.model.feature_importances_)[::-1]
        total = sum(self.model.feature_importances_)
        out = []
        for i in ranking: 
            print((self.feature_names[i], self.model.feature_importances_[i]/total))
            out.append((self.feature_names[i], self.model.feature_importances_[i]/total))
        self.feature_importances = convert_listoftuples_dict(out)
        return self.feature_importances
    def test(self, X_test, y_test, metric=None):
        y_pred = self.model.predict(X_test)
        if not metric: return mean_squared_error(y_test, y_pred) 
        else: return metric(y_test, y_pred)

class ComplexPolynomialFeatures:
    def __init__(self, feature_names, dictionary):
        self.feature_names = feature_names
        self.dictionary = dictionary
        self.poly_feature_names = PolynomialFeatures(include_bias=True).fit(np.array(list(dictionary.values())).squeeze(-1).T.real).get_feature_names(self.feature_names)
        self.output = np.ones(self.dictionary[self.feature_names[0]].shape, dtype=np.complex64)

    def fit(self,):
        for f in self.poly_feature_names[1:]:
            print("Computing", f)
            self.output = np.hstack((self.output, compute_from_description(f, self.dictionary)))
        return self.output

def compute_from_description(description, dictionary, split_keys=(" ", "^")):
    terms = description.split(split_keys[0]); out = 1
    for t in terms:
        if t in dictionary:
            out = out*dictionary[t]
        else:
            t, deg = t.split(split_keys[1])
            out = out*dictionary[t]**int(deg)
    return out

# calculate aic for regression
def calculate_aic(n, mse, num_params):
	aic = n * log(mse) + 2 * num_params
	return aic

def calculate_bic(n, mse, num_params):
	bic = n * log(mse) + num_params * log(n)
	return bic

def occam_razor(scores):
    mse_performances = [e[0] for e in scores]
    complexities = [e[1] for e in scores]
    max_mse = max(mse_performances)
    max_complexity = 10*max(complexities)
    return [-math.log((e[0]/max_mse)/(max_complexity-e[1])) for e in scores]

def pyGRNN_feature_selection(X, y, feature_names):
    IsotropicSelector = FS.Isotropic_selector(bandwidth='rule-of-thumb')
    return IsotropicSelector.feat_selection((X), (y).ravel(), feature_names=feature_names, strategy ='es')

def change_learning_rate(a_optimizer, lr):
    for g in a_optimizer.param_groups:
        g['lr'] = lr
    return a_optimizer

def pcgrad_update(model, model_optimizer, losses):
    updated_grads = []
    
    for i in range(2):
        model_optimizer.zero_grad()
        losses[i].backward(retain_graph=True)

        g_task = []
        for param in model.parameters():
            if param.grad is not None:
                g_task.append(Variable(param.grad.clone(), requires_grad=False))
            else:
                g_task.append(Variable(torch.zeros(param.shape), requires_grad=False))
        # appending the gradients from each task
        updated_grads.append(g_task)

    updated_grads = list(pcgrad.pc_grad_update(updated_grads))[0]
    for idx, param in enumerate(model.parameters()): 
        param.grad = (updated_grads[0][idx]+updated_grads[1][idx])

    return model, model_optimizer, sum(losses)

def create_data_for_feynman(G, target, filename):
    if len(target.shape)>1:
        target = np.squeeze(target) 
    with open(filename, "w") as file:
        for row in range(G.shape[0]):
            string_out = ''
            for col in range(G.shape[1]):
                string_out += str(G[row][col]) + ' '
            string_out += str(target[row])
            if row == G.shape[0]-1:
                file.write(string_out) 
            else:
                file.write(string_out+'\n') 
    print("Done writing into the file")
    file.close()
