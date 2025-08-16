import torch.nn as nn


class Upsample(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x = self.upsample(x)
        return x