import torch.nn as nn 

from .conv import Conv


class C2f(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = Conv(c_in=c_in, c_out=c_out, k=1, s=1, p=0)
        self.bottleneck = BottleNeck(c_in=c_in, c_out=c_out)    

    def forward(self, x):
        x = self.conv(x)
        


class BottleNeck(nn.Module):
    def __init__(self, c_in, c_out, shortcut=True):
        super().__init__()
        self.conv = Conv(c_in=c_in, c_out=c_out, k=3, s=1, p=1)
        self.shortcut = shortcut

    def forward(self, x):
        if self.shortcut:
            x_conv = self.conv(x)
            x_conv = self.conv(x_conv)
            return x + x_conv
        else:
            x = self.conv(x)
            x = self.conv(x)
            return x