import yaml

import torch
import torch.nn as nn


class ModelParser:
    def __init__(self, model_cfg:str):
        self.cfg = yaml.load(model_cfg)

    def parse(self):
        pass

    def make_divisible(self):
