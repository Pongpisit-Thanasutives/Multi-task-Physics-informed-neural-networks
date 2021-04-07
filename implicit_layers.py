import torch
from torch import nn

class TanhNewtonImplicitLayer(nn.Module):
    def __init__(self, out_features, tol = 1e-4, max_iter=50):
        super().__init__()
        self.linear = nn.Linear(out_features, out_features, bias=False)
        self.tol = tol
        self.max_iter = max_iter
  
    def forward(self, x):
        # Run Newton's method outside of the autograd framework
        with torch.no_grad():
            z = torch.tanh(x)
            self.iterations = 0
            while self.iterations < self.max_iter:
                z_linear = self.linear(z) + x
                g = z - torch.tanh(z_linear)
                self.err = torch.norm(g)
                if self.err < self.tol:
                    break

                # newton step
                J = torch.eye(z.shape[1])[None,:,:] - (1 / torch.cosh(z_linear)**2)[:,:,None]*self.linear.weight[None,:,:]
                z = z - torch.solve(g[:,:,None], J)[0][:,:,0]
                self.iterations += 1
    
        # reengage autograd and add the gradient hook
        z = torch.tanh(self.linear(z) + x)
        z.register_hook(lambda grad : torch.solve(grad[:,:,None], J.transpose(1,2))[0][:,:,0])
        return z