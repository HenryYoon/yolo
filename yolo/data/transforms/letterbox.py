import torch
from torchvision.transforms import v2
import torchvision.transforms.v2.functional as F


class LetterBox(v2.Transform):
    def __init__():
        super().__init__()

    def transform(image:torch.Tensor):
        B, C, H, W = image.shape

        image = F.resive(image, max_size=640)

        # move to center

        # padding

        return image
        
