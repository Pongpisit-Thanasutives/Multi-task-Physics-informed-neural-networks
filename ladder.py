import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class LinearLayer(nn.Module):
    def __init__(self, d_in, d_out, bias=False, activation_function=None, noise_std=0.01):
        super(LinearLayer, self).__init__()
        self.linear = nn.Linear(d_in, d_out, bias=bias)
        # weights init using xavier_uniform_
        torch.nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            self.linear.bias.data.fill_(0.01)
        
        self.activation_function = activation_function
        self.noise_std = noise_std
        self.is_clean = 1
        self.buffer = None
        
    def forward(self, h):
        if self.is_clean: return self.forward_clean(h)
        else: return self.forward_noisy(h)
                 
    def forward_clean(self, h):
        z = self.linear(h)
        self.buffer = z.clone()
        if self.activation_function:
            z = self.activation_function(z)
        return z
                 
    def forward_noisy(self, h):
        z = self.linear(h)
        # adding noise
        noise = np.random.normal(loc=0.0, scale=self.noise_std, size=z.size())
        noise = torch.tensor(noise, requires_grad=False).float()
        z = z + noise
        self.buffer = z.clone()
        if self.activation_function:
            z = self.activation_function(z)
        return z
    
    def set_clean(self, is_clean):
        self.is_clean = is_clean

class Encoder(nn.Module):
    def __init__(self, d_in, hidden_dims, d_out, n_layers, bias, activation_function, noise_std):
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.noise_std = noise_std
        # Set a stack of the layers
        self.stacked_layers = nn.Sequential()
        self.stacked_layers.add_module("layer_0", LinearLayer(d_in, hidden_dims, bias, activation_function, noise_std))
        for i in range(n_layers-1):
            self.stacked_layers.add_module("layer_%s" % str(i+1), LinearLayer(hidden_dims, hidden_dims, bias, activation_function, noise_std))
        self.stacked_layers.add_module("layer_%s" % str(n_layers), LinearLayer(hidden_dims, d_out, bias, None, noise_std))
        # Buffer
        self.clean_buffer = None
        self.noisy_buffer = None
        
    def forward(self, h, is_clean=True, is_cache=False):
        # Set the FWD mode 
        for i in range(len(self.stacked_layers)):
            self.stacked_layers[i].set_clean(is_clean)
        
        # FWD
        if is_clean:
            input_tensor = h
            z = self.stacked_layers(input_tensor)
        else:
            noise = np.random.normal(loc=0.0, scale=self.noise_std, size=h.size())
            noise = torch.tensor(noise, requires_grad=False).float()
            input_tensor = h + noise
            z = self.stacked_layers(input_tensor)
        
        # Caching
        if is_cache:
            self.cache_hidden_states(input_tensor, is_clean)
            
        return z
    
    def cache_hidden_states(self, input_tensor, is_clean=True):
        if is_clean:
            self.clean_buffer = [input_tensor]+[self.stacked_layers[i].buffer for i in range(len(self.stacked_layers))]
        else:
            self.noisy_buffer = [input_tensor]+[self.stacked_layers[i].buffer for i in range(len(self.stacked_layers))]

class DecoderLayer(nn.Module):
    def __init__(self, d_in, d_out, bias):
        super(DecoderLayer, self).__init__()
        self.d_in = d_in
        self.d_out = d_out

        self.a1 = Parameter(0. * torch.ones(1, d_in))
        self.a2 = Parameter(1. * torch.ones(1, d_in))
        self.a3 = Parameter(0. * torch.ones(1, d_in))
        self.a4 = Parameter(0. * torch.ones(1, d_in))
        self.a5 = Parameter(0. * torch.ones(1, d_in))

        self.a6 = Parameter(0. * torch.ones(1, d_in))
        self.a7 = Parameter(1. * torch.ones(1, d_in))
        self.a8 = Parameter(0. * torch.ones(1, d_in))
        self.a9 = Parameter(0. * torch.ones(1, d_in))
        self.a10 = Parameter(0. * torch.ones(1, d_in))

        if self.d_out is not None:
            self.V = torch.nn.Linear(d_in, d_out, bias=bias)
            torch.nn.init.xavier_uniform_(self.V.weight)
            if bias:
                self.V.bias.data.fill_(0.01)

        # buffer for hat_z_l to be used for cost calculation
        self.buffer = None

    def g(self, tilde_z_l, u_l):
        ones = Parameter(torch.ones(tilde_z_l.size()[0], 1))

        b_a1 = ones.mm(self.a1)
        b_a2 = ones.mm(self.a2)
        b_a3 = ones.mm(self.a3)
        b_a4 = ones.mm(self.a4)
        b_a5 = ones.mm(self.a5)

        b_a6 = ones.mm(self.a6)
        b_a7 = ones.mm(self.a7)
        b_a8 = ones.mm(self.a8)
        b_a9 = ones.mm(self.a9)
        b_a10 = ones.mm(self.a10)

        mu_l = torch.mul(b_a1, torch.sigmoid(torch.mul(b_a2, u_l) + b_a3)) +                torch.mul(b_a4, u_l) +                b_a5

        v_l = torch.mul(b_a6, torch.sigmoid(torch.mul(b_a7, u_l) + b_a8)) +               torch.mul(b_a9, u_l) +               b_a10

        hat_z_l = torch.mul(tilde_z_l - mu_l, v_l) + mu_l

        return hat_z_l

    def forward(self, tilde_z_l, u_l):
        # hat_z_l will be used for calculating decoder costs
        hat_z_l = self.g(tilde_z_l, u_l)
        # store hat_z_l in buffer for cost calculation
        self.buffer = hat_z_l

        if self.d_out is not None:
            return self.V(hat_z_l)
        else:
            return None

class Decoder(nn.Module):
    def __init__(self, d_in, hidden_dims, d_out, n_layers, bias):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        
        self.stacked_layers = nn.Sequential()
        self.stacked_layers.add_module("layer_0", DecoderLayer(d_in, hidden_dims, bias=bias))
        for i in range(n_layers-1):
            self.stacked_layers.add_module("layer_%s" % str(i+1), DecoderLayer(hidden_dims, hidden_dims, bias=bias))
        self.stacked_layers.add_module("layer_%s" % str(n_layers), DecoderLayer(hidden_dims, d_out, bias=bias))
        
        self.bottom_decoder = DecoderLayer(d_out, None, bias=bias)
        
    def forward(self, tilde_z_states, top):
        # tilde_z_states should be in the reversed order of encoders
        hat_z = []
        for i in range(len(self.stacked_layers)):
            top = self.stacked_layers[i](tilde_z_states[i], top)
            hat_z.append(self.stacked_layers[i].buffer)
        self.bottom_decoder(tilde_z_states[-1], top)
        hat_z.append(self.bottom_decoder.buffer.clone())
        return hat_z

class LadderNetwork(nn.Module):
    def __init__(self, d_in, hidden_dims, d_out, n_layers, bias, activation_function, noise_std):
        super(LadderNetwork, self).__init__()
        encoder_bias, decoder_bias = bias
        self.encoder = Encoder(d_in, hidden_dims, d_out, n_layers, encoder_bias, activation_function, noise_std=noise_std)
        self.decoder = Decoder(d_out, hidden_dims, d_in, n_layers, decoder_bias)
        
    def forward(self, x, include_unsup=True):
        clean_out = self.encoder(x, is_clean=True, is_cache=True)
        ladder_loss = None
        
        if include_unsup:
            noisy_out = self.encoder(x, is_clean=False, is_cache=True)
            
            # decoding, [::-1] => reversing the list
            noisy_buffer = self.encoder.noisy_buffer[::-1]
            decoder_outputs = self.decoder(noisy_buffer, noisy_out)
            ladder_loss = self.unsupervised_loss(decoder_outputs, self.encoder.clean_buffer)
            
        return clean_out, ladder_loss
    
    def unsupervised_loss(self, decoder_outputs, clean_buffer):
        loss = 0.0
        for i in range(len(decoder_outputs)):
            loss += F.mse_loss(clean_buffer[i], decoder_outputs[len(decoder_outputs)-(i+1)])
        return loss

if __name__ == "__main__":
    d_in, hidden_dims, d_out = 2, 50, 1
    bias = False, False
    n_layers = 1 # This counts the number of fully connected layers in a network.
    activation_function = torch.tanh
    noise_std = 0.01

    inpp = torch.rand(100, 2)
    
    network = LadderNetwork(d_in=d_in, hidden_dims=hidden_dims, 
                d_out=d_out, n_layers=n_layers, bias=bias, 
                activation_function=activation_function, noise_std=noise_std)

    u, unsup_loss = network(inpp)

    print(u)
    print(unsup_loss)
    print("Test passing")
