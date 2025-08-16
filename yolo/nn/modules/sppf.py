import torch.nn as nn 

from .conv import Conv
from .concat import Concat


class SPPF(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = Conv(c_in=c_in, c_out=c_out, k=1, s=1, p=0)
        self.pool = nn.MaxPool2d(kernel_size=1, stride=1)
        self.concat = Concat()


    def forward(self, x):
        x_conv = self.conv(x)
        
        x_max1 = self.pool(x)
        x_max2 = self.pool(x_max1)
        x_max3 = self.pool(x_max2)

        x_concat = [x_conv, x_max1, x_max2, x_max3]

        x = self.concat(x_concat)
        x = self.conv(x)
        return x