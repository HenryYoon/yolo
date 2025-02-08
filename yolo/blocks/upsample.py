import torch
import torch.nn as nn 

class Upsample(nn.Module):
    def __init__(self):
        self.upsample = nn.Upsample(None, 2, "nearest")
    
    def forward(self, x):
        return self.upsample(x)