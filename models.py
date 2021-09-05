import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable, grad

import sympytorch
from complexPyTorch.complexLayers import ComplexBatchNorm1d, ComplexDropout, ComplexLinear
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d

from cplxmodule import nn as cnn
from cplxmodule import cplx
from cplxmodule.nn import RealToCplx, CplxToReal, CplxToCplx, CplxSequential
from cplxmodule.nn import CplxLinear, CplxModReLU, CplxDropout, CplxBatchNorm1d

from utils import diff_flag, to_complex_tensor, dimension_slicing, string2sympytorch, gradients_dict, build_exp

from tqdm import trange

def cat(*args): return torch.cat(args, dim=-1)

def to_column_vector(arr):
    return arr.flatten()[:, None]

def cplx2tensor(func):
    return func.real + 1j*func.imag

def add_imaginary_dimension(a_tensor):
    return torch.hstack([a_tensor, torch.zeros(a_tensor.shape[0], 1).requires_grad_(False)])

def real2cplx(real_tensor):
    if real_tensor.dtype == torch.complex64 or real_tensor.dtype == torch.complex32: ct = real_tensor
    else: ct = torch.complex(real_tensor, torch.zeros_like(real_tensor))

    out = []
    for te in dimension_slicing(ct):
        out.append(te.real)
        out.append(te.imag)

    return RealToCplx()(cat(*out))

def complex_mse(v1, v2, dist_fn=F.mse_loss):
    return dist_fn(v1.real, v2.real) + dist_fn(v1.imag, v2.imag)

def real_mse(v1, v2):
    row = min(v1.shape[0], v2.shape[0])
    return F.mse_loss(v1[:row, :], v2[:row, :])

def diff(func, inp):
    return grad(func, inp, create_graph=True, retain_graph=True, allow_unused=True, grad_outputs=torch.ones(func.shape, dtype=func.dtype))[0]

class ImaginaryDimensionAdder(nn.Module):
    def __init__(self,):
        super(ImaginaryDimensionAdder, self).__init__(); pass
    def forward(self, real_tensor):
        added = cat(real_tensor[:, 0:1], torch.zeros(real_tensor.shape[0], 1))
        for i in range(1, real_tensor.shape[1]):
            added = cat(added, real_tensor[:, i:i+1], torch.zeros(real_tensor.shape[0], 1))
        return added

class TorchMLP(nn.Module):
    def __init__(self, dimensions, bias=True, activation_function=nn.Tanh(), bn=None, dropout=None):
        super(TorchMLP, self).__init__()
        self.model  = nn.ModuleList()

        for i in range(len(dimensions)-1):
            self.model.append(nn.Linear(dimensions[i], dimensions[i+1], bias=bias))
            if bn is not None and i!=len(dimensions)-2:
                self.model.append(bn(dimensions[i+1]))
                if dropout is not None:
                    self.model.append(dropout)
            if i==len(dimensions)-2: break
            self.model.append(activation_function)

    def xavier_init(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        for i, l in enumerate(self.model): 
            x = l(x)
        return x

class ComplexTorchMLP(nn.Module):
    def __init__(self, dimensions, bias=True, activation_function=CplxToCplx[torch.tanh](), bn=False, dropout_rate=0.0):
        super(ComplexTorchMLP, self).__init__()
        self.model  = [] 
        self.bias = bias
        self.dropout = None
        if dropout_rate>0.0: self.dropout = CplxDropout
        else: self.dropout = None
        if bn: self.bn = CplxBatchNorm1d
        else: self.bn = None
        for i in range(len(dimensions)-1):
            linear = CplxLinear(dimensions[i], dimensions[i+1], bias=self.bias)
            self.model.append(linear)
            if self.bn is not None and i!=len(dimensions)-2:
                self.model.append(self.bn(dimensions[i+1]))
                if self.dropout is not None:
                    self.model.append(self.dropout(dropout_rate))
            if i==len(dimensions)-2: break
            self.model.append(activation_function)
        self.model = CplxSequential(*self.model) 

    def xavier_init(self, m):
        if type(m) == nn.Linear or type(m) == CplxLinear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        for i, l in enumerate(self.model): 
            x = l(x)
        return x

class TorchComplexMLP(nn.Module):
    def __init__(self, dimensions, bias=True, activation_function=nn.Tanh(), bn=None, dropout_rate=0.0):
        super(TorchComplexMLP, self).__init__()
        print("This class is deprecated.")
        print("The implementation was based on complexPyTorch, which will be no longer used.")
        self.model  = nn.ModuleList()
        self.dropout = None
        if dropout_rate>0.0: 
            self.dropout = ComplexDropout(dropout_rate)
        self.bn = bn

        for i in range(len(dimensions)-1):
            linear = ComplexLinear(dimensions[i], dimensions[i+1])
            linear.fc_r.apply(self.xavier_init)
            linear.fc_i.apply(self.xavier_init)
            self.model.append(linear)
            if self.bn is not None and i!=len(dimensions)-2:
                self.model.append(self.bn(dimensions[i+1]))
                if self.dropout is not None:
                    self.model.append(self.dropout)
            if i==len(dimensions)-2: break
            self.model.append(activation_function)

        try:
            self.model.apply(self.xavier_init)
        except:
            print("Cannot init the complex networ")
            pass

    def xavier_init(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        for i, l in enumerate(self.model): 
            x = l(x)
        return x

class Network(nn.Module):
    def __init__(self, model, index2features = ('uf', 'u_x',  'u_xx', 'u_tt', 'u_xt', 'u_tx'), scale=False, lb=None, ub=None):
        super(Network, self).__init__()
        # pls init the self.model before
        self.model = model
        # For tracking
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

    def get_selector_data_old(self, x, t):
        uf = self.forward(x, t)

        ### PDE Loss calculation ###
        # first-order derivatives
        u_t = self.gradients(uf, t)[0]
        u_x = self.gradients(uf, x)[0]
        # Homo second-order derivatives
        u_tt = self.gradients(u_t,t)[0]
        u_xx = self.gradients(u_x, x)[0]
        # Hetero second-order derivatives
        u_xt = self.gradients(u_t, x)[0]
        u_tx = self.gradients(u_x, t)[0]

        X_selector = torch.cat([uf, u_x, u_xx, u_tt, u_xt, u_tx], dim=1)
        y_selector = u_t

        return X_selector, y_selector

    def get_selector_data(self, x, t):
        uf = self.forward(x, t)
        u_t = self.gradients(uf, t)[0]
        
        ### PDE Loss calculation ###
        # Without calling grad
        derivatives = []
        for t in self.diff_flag[0]:
            if t=='uf': derivatives.append(uf)
            elif t=='x': derivatives.append(x)
        # With calling grad
        for t in self.diff_flag[1]:
            out = uf
            for c in t:
                if c=='x': out = self.gradients(out, x)[0]
                elif c=='t': out = self.gradients(out, t)[0]
            derivatives.append(out)
        
        return torch.cat(derivatives, dim=1), u_t

    def gradients(self, func, x):
        return grad(func, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones(func.shape))

    def neural_net_scale(self, inp): 
        return 2*(inp-self.lb/(self.ub-self.lb))-1

class SeclectorNetwork(nn.Module):
    def __init__(self, X_train_dim, bn=None):
        super(SeclectorNetwork, self).__init__()
        # Nonlinear model, Training with PDE reg.
        self.nonlinear_model = TorchMLP(dimensions=[X_train_dim, 50, 50, 1], activation_function=nn.Tanh(), bn=bn, dropout=nn.Dropout(p=0.1))
        
    def xavier_init(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        
    def forward(self, inn):
        ut_approx = self.nonlinear_model(inn)
        return ut_approx
    
    def loss(self, X_input, y_input):
        ut_approx = self.forward(X_input)
        mse_loss = F.mse_loss(ut_approx, y_input, reduction='mean')
        return mse_loss

class AttentionSelectorNetwork(nn.Module):
    def __init__(self, layers, prob_activation=torch.sigmoid, bn=None, reg_intensity=0.3):
        super(AttentionSelectorNetwork, self).__init__()
        # Nonlinear model, Training with PDE reg.
        assert len(layers) > 1
        self.linear1 = nn.Linear(layers[0], layers[0])
        self.prob_activation = prob_activation
        self.nonlinear_model = TorchMLP(dimensions=layers, activation_function=nn.Tanh(), bn=bn, dropout=nn.Dropout(p=0.1))
        self.latest_weighted_features = None
        self.th = 0.5
        self.reg_intensity = reg_intensity
        
    def xavier_init(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        
    def forward(self, inn):
        return self.nonlinear_model(inn*self.weighted_features(inn))
    
    def weighted_features(self, inn):
        self.latest_weighted_features = self.prob_activation(self.linear1(inn)).mean(axis=0)
        return self.latest_weighted_features
    
    def loss(self, X_input, y_input):
        ut_approx = self.forward(X_input)
        mse_loss = F.mse_loss(ut_approx, y_input, reduction='mean')
        return mse_loss+self.reg_intensity*torch.norm(F.relu(self.latest_weighted_features-self.th), p=0)

        return self.network.uf, unsup_loss

class SemiSupModel(nn.Module):
    def __init__(self, network, selector, normalize_derivative_features=False, mini=None, maxi=None):
        super(SemiSupModel, self).__init__()
        self.network = network
        self.selector = selector
        self.normalize_derivative_features = normalize_derivative_features
        self.mini = mini
        self.maxi = maxi
    def forward(self, X_u_train, scale=False):
        X_selector, y_selector = self.network.get_selector_data(*dimension_slicing(self.neural_net_scale(X_u_train, self.lb, self.ub)))
        if self.normalize_derivative_features:
            X_selector = (X_selector-self.mini)/(self.maxi-self.mini)
        unsup_loss = self.selector.loss(X_selector, y_selector)
        return self.network.uf, unsup_loss

# Using uncerts to weight each PDE loss function
class UncertaintyWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = UncertaintyWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(UncertaintyWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = []
        for i, loss in enumerate(x):
            loss_sum.append(0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2))
        return loss_sum

class TanhProb(nn.Module):
    def __init__(self,):
        super(TanhProb, self).__init__()
        self.act = nn.Tanh()
    def forward(self, X):
        return 0.5*(self.act(X)+1.0)

# My version of sympytorch.SymPyModule
class SympyTorch(nn.Module):
    def __init__(self, expressions):
        super(SympyTorch, self).__init__()
        self.mod = sympytorch.SymPyModule(expressions=expressions)
    def forward(self, gd):
        return torch.squeeze(self.mod(**gd), dim=-1)

# Extension of basic sympymodule for supporting operations with complex numbers
class ComplexSymPyModule(nn.Module):
    def __init__(self, expressions, complex_coeffs=None, learnable_parts=[True, True]):
        super(ComplexSymPyModule, self).__init__()
        self.sympymodule = sympytorch.SymPyModule(expressions=expressions)
        if complex_coeffs is None: 
            self.reals = nn.Parameter(torch.rand(len(expressions), 1))
            self.imags = nn.Parameter(torch.rand(len(expressions), 1))
        else:
            complex_tensor_coeffs = to_complex_tensor(complex_coeffs)
            self.reals = nn.Parameter(complex_tensor_coeffs.real.reshape(-1, 1))
            self.imags = nn.Parameter(complex_tensor_coeffs.imag.reshape(-1, 1))
        if learnable_parts[0] == False:
            self.reals.requires_grad_(False)
        if learnable_parts[1] == False:
            self.imags.requires_grad_(False)
                    
    def forward(self, kwargs):
        return (torch.squeeze(self.sympymodule(**kwargs)).type(torch.complex64)@self.complex_coeffs()).reshape(-1, 1)
    def complex_coeffs(self,):
        return torch.complex(self.reals, self.imags)

# not expect n_inputs != 1
class CoeffLearner(nn.Module):
    def __init__(self, init_data=None):
        super(CoeffLearner, self).__init__()
        if init_data is None: init_data = torch.rand(self.n_inputs, requires_grad=True)
        self.coeffs = nn.Parameter(data=torch.tensor(init_data).float(), requires_grad=True)

    def forward(self, X):
        return self.coeffs*X

class PartialDerivativeCalculator(nn.Module):
    def __init__(self, expressions, funcs, trainable_one=True):
        super(PartialDerivativeCalculator, self).__init__()
        mvs = [string2sympytorch(e) for e in expressions]
        self.mds = nn.ModuleList([e[0] for e in mvs])
        self.variables = [e[1] for e in mvs]
        self.n_vars = len(self.variables)
        self.variables = [sorted(list(map(str, self.variables[i]))) for i in range(self.n_vars)]

        # Functions depend on (x, t)
        self.funcs = nn.ModuleList()
        self.funcs_variables = []
        for s in funcs:
            expr, var = build_exp(s, trainable_one=trainable_one)
            if len(var) > 0: self.funcs.append(SympyTorch(expressions=[expr]))
            elif len(var) == 0: self.funcs.append(CoeffLearner(init_data=float(s)))
            else: print("Error")
            self.funcs_variables.append(list(map(str, var)))

    def forward(self, u, x, t):
        out = 0.0
        for i in range(self.n_vars):
            computed = self.mds[i](gradients_dict(u, x, t, self.variables[i]))
            feed_dict = {}
            for e in self.funcs_variables[i]: feed_dict[e] = eval(e) 
            if len(feed_dict) > 0: computed = computed*self.funcs[i](feed_dict)
            elif len(feed_dict) == 0: computed = self.funcs[i](computed)
            else: print("Error")
            out = out+computed
        return out


class CancelOut(nn.Module):
    def __init__(self, n_features, *kargs, **kwargs):
        super(CancelOut, self).__init__()
        self.weights = nn.Parameter(torch.zeros(n_features, requires_grad=True) + 4)
    def forward(self, x):
        return (x * torch.sigmoid(self.weights.float()))
   
# return reconstruction error + KL divergence losses
def vae_loss(recon_x, x, mu, log_var):
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

class VAE(nn.Module):
    def __init__(self, x_dim=2, h_dim=32, z_dim=2):
        super(VAE, self).__init__()
        
        # encoder part
        self.fc = nn.Linear(x_dim, h_dim)
        self.ec_mu = nn.Linear(h_dim, z_dim)
        self.ec_logvar = nn.Linear(h_dim, z_dim)
        # decoder part
        self.dc1 = nn.Linear(z_dim, h_dim)
        self.dc2 = nn.Linear(h_dim, x_dim)
        
    def encoder(self, x):
        h = F.relu(self.fc(x))
        return self.ec_mu(h), self.ec_logvar(h) # mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z):
        h = F.relu(self.dc1(z))
        h = self.dc2(h)
        return h
    
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

def ae_loss(recon_X, X, include_l1=torch.FloatTensor([0.0]), reduction="mean"):
    output_loss = F.mse_loss(recon_X, X, reduction=reduction)
    if include_l1 > 0.0: output_loss = output_loss + torch.abs(include_l1)*F.l1_loss(recon_X, X, reduction=reduction) 
    return output_loss

def complex_ae_loss(recon_X, X, include_l1=torch.FloatTensor([0.0])):
    output_loss = complex_mse(recon_X, X)
    if include_l1 > 0.0: output_loss = output_loss + torch.abs(include_l1)*complex_mse(recon_X, X, dist_fn=F.l1_loss)
    return output_loss

class AutoEncoder(nn.Module):
    def __init__(self, x_dim=2, h_dim=32, activation=nn.ReLU(), include_l1=0.1):
        super(AutoEncoder, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(x_dim, h_dim), activation, nn.Linear(h_dim, x_dim))
        self.l1_strength = None
        if include_l1 > 0.0:
            self.l1_strength = nn.Parameter(data=torch.FloatTensor([include_l1]), requires_grad=True)

    def forward(self, X, split=False):
        if split: return dimension_slicing(self.mlp(X))
        return self.mlp(X)

    def compute_loss(self, X, reduction="mean"):
        recon_X = self.mlp(X)
        output_loss = F.mse_loss(recon_X, X, reduction=reduction)
        return output_loss + torch.abs(self.l1_strength)*F.l1_loss(recon_X, X, reduction=reduction) 

    def pretrain(self, X, epochs=10000, reduction="mean"):
        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        for i in trange(epochs):
            opt.zero_grad()
            self.compute_loss(X, reduction=reduction).backward(retain_graph=True)
            opt.step()
        print("Loss:", self.test(X))
        print("Done pretraining")
        return self

    def test(self, X):
        self.eval()
        return F.mse_loss(self(X), X).item()

class ComplexAutoEncoder(nn.Module):
    def __init__(self, x_dim=2, h_dim=32, activation=CplxModReLU(), include_l1=0.1):
        super(ComplexAutoEncoder, self).__init__()
        self.mlp = CplxSequential(CplxLinear(x_dim, h_dim), activation, CplxLinear(h_dim, x_dim))
        self.l1_strength = None
        if include_l1 > 0.0:
            self.l1_strength = nn.Parameter(data=torch.FloatTensor([include_l1]), requires_grad=True)
    
    def forward(self, X, split=False):
        if split: return dimension_slicing(self.mlp(X))
        return self.mlp(X)

    def compute_loss(self, X):
        recon_X = self.forward(X)
        output_loss = complex_mse(recon_X, X)
        return output_loss + torch.abs(self.l1_strength)*complex_mse(recon_X, X, dist_fn=F.l1_loss)

    def pretrain(self, X, epochs=10000):
        if type(X) == torch.Tensor: X = real2cplx(X)
        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.train()
        for i in trange(epochs):
            opt.zero_grad()
            self.compute_loss(X).backward(retain_graph=True)
            opt.step()
        print("Loss:", self.test(X))
        print("Done pretraining")
        return self

    def test(self, X):
        self.eval()
        if type(X) == torch.Tensor: X = real2cplx(X)
        return complex_mse(self(X), X).item()

class RobustPCANN(nn.Module):
    def __init__(self, beta=0.0, is_beta_trainable=True, inp_dims=2, hidden_dims=50):
        super(RobustPCANN, self).__init__()
        if is_beta_trainable: self.beta = nn.Parameter(data=torch.FloatTensor([beta]), requires_grad=True)
        else: self.beta = beta
        self.proj = nn.Sequential(nn.Linear(inp_dims, hidden_dims), nn.Tanh(), nn.Linear(hidden_dims, inp_dims), nn.Tanh())

    def forward(self, O, S, order="fro", normalize=True, is_clamp=True):
        corr = self.proj(S)
        if normalize: corr = corr / torch.norm(corr, p=order)
        if is_clamp: beta = torch.clamp(self.beta, min=-1.0, max=1.0)
        else: beta = self.beta
        return O - beta*corr
