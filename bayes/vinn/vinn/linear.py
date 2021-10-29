import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.distributions import Normal
import math

from . import Module
from .utils import kl_divergence

class Linear(Module):

    def __init__(
        self, 
        in_features: int, 
        out_features: int,
        posterior = Normal,
        prior = Normal(0, 1),
        bias: bool = True,
        loc_rep: bool = True
        ) -> None:
        
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_loc = Parameter(torch.Tensor(out_features, in_features))
        self.weight_ro = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias_loc = Parameter(torch.Tensor(out_features))
            self.bias_ro = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_loc', None)
            self.register_parameter('bias_ro', None)
        self.loc_rep = loc_rep
        self.prior = prior
        self.posterior = posterior
        self.reset_parameters()

    @property
    def weight_scale(self):
        return F.softplus(self.weight_ro)

    @property
    def bias_scale(self):
        if self.bias_ro is None:
            return None
        return F.softplus(self.bias_ro)

    @property
    def weight_posterior(self):
        return self.posterior(self.weight_loc, self.weight_scale)

    @property
    def bias_posterior(self):
        return self.posterior(self.bias_loc, self.bias_scale)

    def weight_sample(self):
        return self.weight_posterior.rsample()

    def bias_sample(self):
        if self.bias_loc is None:
            return None
        return self.bias_posterior.rsample()

    @property
    def kl(self):
        kl_div = kl_divergence(self.weight_posterior, self.prior).sum()
        if self.bias_loc is not None:
            kl_div += kl_divergence(self.bias_posterior, self.prior).sum()
        return kl_div

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight_loc, a=math.sqrt(5))
        init.constant_(self.weight_ro, -3)        
        if self.bias_loc is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_loc)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_loc, -bound, bound)
            init.constant_(self.bias_ro, -3)

    def forward(self, input):
        if self.loc_rep:
            output_loc = F.linear(input, self.weight_loc, self.bias_loc)
            output_scale = torch.sqrt(1e-9 + F.linear(input.pow(2), self.weight_scale.pow(2), self.bias_scale.pow(2)))
            return self.posterior(output_loc, output_scale).rsample()
        else:
            return F.linear(input, self.weight_sample(), self.bias_sample())

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias_loc is not None
        )