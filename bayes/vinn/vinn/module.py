import torch.nn as nn

class Module(nn.Module):
   
    def __init__(self):
        super(Module, self).__init__()       
    
    @property
    def kl(self):
        kl = 0
        for submodule in self.modules():
            if submodule == self:
                continue
            if hasattr(submodule, 'kl'):
                kl += submodule.kl
        return kl