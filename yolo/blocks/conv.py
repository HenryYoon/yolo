import torch
import torch.nn as nn 


class Conv(nn.Module):
    def __init__(self, k, s, p, c):
        self.conv = Conv2d()
        self.bn = BatchNorm2d()
        self.act = SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = act(x)
        return x