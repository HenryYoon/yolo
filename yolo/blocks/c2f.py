import torch
import torch.nn as nn 


class C2f(nn.Module):
    def __init__(self, c_in, c_out, n):
        self.conv = Conv(k=1, s=1, p=0, c=c_out)
        self.bottleneck = nn.ModuleList([
            Bottleneck() for _ in n
        ])
        self.concat = Concat()
        self.split = Split()

    def forward(self,x):
        x = self.conv(x)
        x_concat1, x_concat2 = self.split(x)
        x_concat3 = self.bottleneck(x_split2)
        x = self.concat([x_concat1, x_concat2, x_concat3])
        x = self.conv(x)
        return x