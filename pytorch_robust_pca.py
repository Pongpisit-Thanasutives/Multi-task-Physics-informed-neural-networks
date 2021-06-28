import torch
from torch import nn
import torch.nn.functional as F
from torch.linalg import norm as Norm

# This module has 0 param, and act like a function
class R_pca(nn.Module):
    def __init__(self, D, mu=None, lmbda=None, verbose=False):
        super(R_pca, self).__init__()
        self.D = D
        self.S = torch.zeros(self.D.shape)
        self.Y = torch.zeros(self.D.shape)

        if mu is not None: self.mu = mu
        else: self.mu = torch.prod(torch.tensor(self.D.shape)) / (4*torch.linalg.norm(self.D, ord=1))
            
        self.mu_inv = 1 / self.mu

        if lmbda is not None: self.lmbda = lmbda
        else: self.lmbda = 1 / torch.sqrt(torch.tensor(self.D.shape).max())
            
        self.verbose = verbose

    def frobenius_norm(self, M):
        return torch.linalg.norm(M, ord='fro')

    def shrink(self, M, tau):
        return torch.sign(M)*torch.maximum((torch.abs(M) - tau), torch.zeros(M.shape))

    def svd_threshold(self, M, tau):
        U, S, V = torch.linalg.svd(M, full_matrices=False)
        return U@(torch.diag(self.shrink(S, tau))@V)

    def forward(self, tol=None, max_iter=1000, iter_print=100):
        iter = 0
        err = np.Inf
        Sk = self.S
        Yk = self.Y
        Lk = torch.zeros(self.D.shape)

        if tol: _tol = tol
        else: _tol = 1E-7 * self.frobenius_norm(self.D)

        # this loop implements the principal component pursuit (PCP) algorithm
        # located in the table on page 29 of https://arxiv.org/pdf/0912.3599.pdf
        while (err > _tol) and iter < max_iter:
            Lk = self.svd_threshold(
                self.D - Sk + self.mu_inv * Yk, self.mu_inv)                            #this line implements step 3
            Sk = self.shrink(
                self.D - Lk + (self.mu_inv * Yk), self.mu_inv * self.lmbda)             #this line implements step 4
            Yk = Yk + self.mu * (self.D - Lk - Sk)                                      #this line implements step 5
            err = self.frobenius_norm(self.D - Lk - Sk)
            iter += 1
            
            # For debugging
            if self.verbose:
                if (iter % iter_print) == 0 or iter == 1 or iter > max_iter or err <= _tol:
                    print('iteration: {0}, error: {1}'.format(iter, err))

        self.L = Lk
        self.S = Sk
        
        return Lk, Sk

class RobustPCANeuralNet(nn.Module):
    def __init__(self, input_tensor, mu=None, lmbda=None):
        super(RobustPCANeuralNet, self).__init__()

        self.input_shape = input_tensor.shape
        if mu is not None: self.mu = mu
        else: self.mu = torch.prod(torch.tensor(self.input_shape)) / (4*Norm(input_tensor, ord=1))
        self.inv_mu = 1.0/self.mu

        if lmbda is not None: self.lmbda = lmbda
        else: self.lmbda = 1 / torch.sqrt(torch.tensor(self.input_shape).max())

        print("The settings are ...")
        print("Lambda:", self.lmbda)
        print("Inverse mu:", self.inv_mu)

        self.S = nn.Parameter(data=torch.randn(self.input_shape).float(), requires_grad=True)

    def forward(self, M):
        return M - self.S
    
    # Call this only after calling the forward function
    # Loss is nan -> if torch.sum(torch.isnan(model.S.grad)) > 0: opt.zero_grad(); break
    def loss(self, M, L):
        return torch.linalg.matrix_norm(L, ord='nuc') + self.lmbda*Norm(self.S, ord=1) + self.inv_mu*Norm(M-L-self.S, ord='fro')

if __name__ == "__main__":
    import numpy as np

    # generate low rank synthetic data
    N = 100
    num_groups = 3
    num_values_per_group = 40
    p_missing = 0.2

    Ds = []
    for k in range(num_groups):
        d = np.ones((N, num_values_per_group)) * (k + 1) * 10
        Ds.append(d)

    D = np.hstack(Ds)

    # decimate 20% of data
    n1, n2 = D.shape
    S = np.random.rand(n1, n2)
    D[S < 0.2] = 0

    D = torch.tensor(D).requires_grad_(True)

    rpca = R_pca(D, verbose=False)
    L, S = rpca(max_iter=10000, iter_print=100)

    print('MSE Loss:', F.mse_loss(D, L+S).item())
