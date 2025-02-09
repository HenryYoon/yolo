import torch
import torch.nn as nn
import yaml
from collections import OrderedDict


class YOLOModel(nn.Module):
    def __init__(self, cfg_path, num_classes=80):
        super(YOLOModel, self).__init__()
        self.num_classes = num_classes
        self.layers = self.parse_model(cfg_path)


    def forward(self, x):
        outputs = {}
        for i, (layer, from_ids) in enumerate(self.layers):
            x = [outputs[f] for f in from_ids] if isinstance(from_ids, list) else outputs[from_ids]
            outputs[i] = layer(x)
        return outputs[len(outputs) - 1]


    def build_layer(self, module_name, args):
        """
        getattr()을 활용하여 `layers/` 폴더에서 동적으로 레이어 클래스를 가져옴.
        """
        try:
            module_cls = getattr(__import__("yolo/blocks", fromlist=[module_name]), module_name)
            return module_cls(*args)
        except AttributeError:
            raise ValueError(f"Unknown module type: {module_name}")


    def parse_model(self, cfg_path):
        """
        YAML 파일을 파싱하여 네트워크 구조를 생성하는 함수.
        """
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)

        layers = OrderedDict()

        for i, (from_idx, repeats, module_name, args) in enumerate(cfg["backbone"] + cfg["head"]):
            layer = self.build_layer(module_name, args)
            if repeats > 1:
                layer = nn.Sequential(*[self.build_layer(module_name, args) for _ in range(repeats)])
            layers[i] = (layer, from_idx)

        return list(layers.items())