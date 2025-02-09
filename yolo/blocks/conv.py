import torch
import torch.nn as nn 

import math


class Conv(nn.Module):
    def __init__(self, in_channel, k, s, p, c, out_channel):
        super().__init__()
        out_channel = self.make_divisible(out_channel, divisor=8)
        p = self.autopad(kernel=k, padding=p)

        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel=k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = nn.SiLU()


    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
    

    def make_divisible(self, x, divisor):
        return math.ceil(x / divisor) * divisor
    
    
    def autopad(kernal, padding=None, dilation=1):  # kernel, padding, dilation
        """Pad to 'same' shape outputs."""
        if dilation > 1:
            kernal = dilation * (kernal - 1) + 1 if isinstance(kernal, int) else [dilation * (x - 1) + 1 for x in kernal]  # actual kernel-size
        if padding is None:
            padding = kernal // 2 if isinstance(kernal, int) else [x // 2 for x in kernal]  # auto-pad
        return padding