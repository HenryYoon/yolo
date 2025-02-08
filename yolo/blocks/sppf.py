import torch
import torch.nn as nn


class SPPF(nn.Module):
    def __init__(self):
        self.conv = Conv(k=1, s=1, p=0)
        self.maxpool = MaxPool2d()
        self.concat = Concat()

    def forward(self, x):
        x1 = self.conv(x)
        x_max1 = self.maxpool(x1)
        x_max2 = self.maxpool(x_max1)
        x_max3 = self.maxpool(x_max2)

        x = self.concat([x1, x_max1, x_max2, x_max3])
        x = self.conv(x)

        return x
        