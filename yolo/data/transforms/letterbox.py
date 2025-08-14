import torch
from torchvision.transforms import v2
import torchvision.transforms.v2.functional as F


class LetterBox(v2.Transform):
    def __init__():
        super().__init__()

    def transform(image:torch.Tensor):
        B, C, H, W = image.shape

        image = F.resize(image, max_size=640)

        # padding
        gain = min(H / 640, W / 640)
        new_unpad = (int(round(W*gain)), int(round(H*gain)))

        dw, dh = 640 - new_unpad[0], 640 - new_unpad[1]
        dw /= 2; dh /= 2

        top, left = int(round(dh-0.1)), int(round(dw-0.1))
        bottom, right = int(round(dh+0.1)), int(round(dw+0.1))

        image = F.pad(image, [left, top, right, bottom], fill=114)

        return image
        
