import torch
from torch import nn

import torchinfo


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 1, (3, 3), padding=1),
            nn.GELU(),
            nn.Conv2d(1, 1, (3, 3), padding=1),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.net(x)
        return x


net = Network()

torchinfo.summary(net, (100, 1, 100, 100))
