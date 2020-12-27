import torch.nn as nn
from torch.nn import Parameter, functional as F

import math

import torch
import torch.nn as nn
import numpy as np

class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, 11, 4, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(64, 192, 5, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(192, 384, 3, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

    def forward(self, x):
        x = self.layers(x)
        x = self.avgpool(x)
        return torch.flatten(x,1)
