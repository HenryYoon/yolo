import torch 
import torch.nn as nn 


class Conv(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=2, p=1):
        super().__init__()
        self.conv = nn.Conv2d(kernel_size=k, stride=s, padding=p, in_channels=c_in, out_channels=c_out)
        self.bn = nn.BatchNorm2d()
        self.act = nn.SiLU()


    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x