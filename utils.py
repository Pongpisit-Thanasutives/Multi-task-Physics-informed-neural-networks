import pickle
from sympy import *
from sympy.parsing.sympy_parser import parse_expr
from sympy.core import evaluate

### Model-related imports ###
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

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
    x = Symbol("x"); y = Symbol("y")
    
    local_dict = {
        "add": Add,
        "mul": Mul,
        "sub": Lambda((x, y), x - y),
        "div": Lambda((x, y), x/y),
    }
    
    exp = simplify(parse_expr(str(program), local_dict=local_dict))
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

def dimension_slicing(a_tensor):
    c = a_tensor.shape[-1]
    out = []
    for i in range(1, c+1): out.append(a_tensor[:, i-1:i])
    return out

def is_nan(a_tensor):
    return torch.isnan(a_tensor).any().item()

def get_dataloader(X_train, y_train, bs):
    return DataLoader(TrainingDataset(X_train, y_train), batch_size=bs)

class TrainingDataset(Dataset):
    def __init__(self, X_train, y_train):
        super(TrainingDataset, self).__init__()
        self.X = X_train
        self.y = y_train
        
    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.X.shape[0]
    
class LadderLoss(nn.Module):
    def __init__(self,):
        super().__init__()
        
    def forward(self, outputs, labels):
        return F.mse_loss(outputs[0], labels) + outputs[1]

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
