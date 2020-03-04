import torch
from torch import nn

class LSGAN(nn.Module):
    def __init__(self):
        super(LSGAN, self).__init__()

    def forward(self, target, is_real):
        if is_real:
            loss = (1 - target)**2
        else:
            loss = target**2
        return torch.mean(loss)

class WeightedL1Loss(nn.Module):
    def __init__(self):
        super(WeightedL1Loss, self).__init__()
    def forward(self, input, target, weight):
        return torch.mean(weight * torch.abs(input - target))
