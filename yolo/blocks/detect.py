import torch
import torch.nn as nn 

class Detect(nn.Module):
    def __init__(self, c):
        self.conv = Conv(k=1, s=1, p=1)
        self.conv2d = Conv2d(k=1, s=1, p=1, c=c)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv(x)
        x = self.conv2d(x)
        return x