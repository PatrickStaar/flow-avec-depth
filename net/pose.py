from torch import nn
import torch.nn.functional as F
import torchvision
from collections import OrderedDict
from .utils import *


class Pose(nn.Module):
    def __init__(self):

        super(Pose, self).__init__()
        
        self.conv_block = nn.Sequential(
            conv(2048, 1024,activation='relu'),
            conv(1024, 512,activation='relu'),
            conv(512, 256,activation='relu'),
        )
        self.adaptive_pooling = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Linear(256, 6)
        )

    def forward(self, features):
        p = self.conv_block(features[-1])
        p = p.mean(3).mean(2)
        # p = self.adaptive_pooling(p)
        p = th.flatten(p, start_dim=1)
        p = self.fc(p)
        # p = F.tanh(p)
        p = 0.01*p
        return p