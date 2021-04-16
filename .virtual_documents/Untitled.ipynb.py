import torch
from torch import nn as nn


class ThresholdSoftmax(nn.Module):
   def __init__(self, th=0.5):
        super().__init__()
        self.th = nn.Parameter(torch.FloatTensor([th]))
        self.sm = nn.Softmax(dim=-1)
   def forward(self, inn):
        imm = self.sm(inn)
        imm = torch.where(imm > self.th, imm, torch.FloatTensor([0.0]))
        return imm


af = ThresholdSoftmax()


x = torch.FloatTensor([1,2,3])
x.requires_grad_(True)
x


(af(x))





x = torch.randn(3, 2)


torch.where(x > 0, x, 0.)



