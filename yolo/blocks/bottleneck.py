import torch
import torch.nn as nn 

class Bottleneck(nn.Module):
    def __init__(self, shortcut:bool=True):
        self.conv = Conv(k=3, s=1, p=1)
        self.shortcut = shortcut

    def forward(self, x):
        x1 = self.conv(x)
        x = self.conv(x1)
        if shortcut:
            return x + x1
        else:
            return x