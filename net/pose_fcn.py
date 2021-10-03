from torch import nn
from .utils import *
from . import resnet18


class Pose(nn.Module):
    def __init__(self):

        super(Pose, self).__init__()
        self.encoder=resnet18(input_channels=6, no_top=True)
        self.conv_block = nn.Sequential(
            conv(512, 512,stride=1,activation='relu'),
            conv(512, 256,stride=1,activation='relu'),
            conv(256, 6,stride=1,output=True),
        )
        # self.fc = nn.Sequential(
        #     nn.Linear(256, 256),
        #     nn.LeakyReLU(0.1,inplace=True),
        #     nn.Linear(256, 256),
        #     nn.LeakyReLU(0.1,inplace=True),
        #     nn.Linear(256, 6)
        # )

    def forward(self, inputs):
        x=self.encoder(inputs)
        p = self.conv_block(x)
        p = p.mean(3).mean(2)
        # p = self.adaptive_pooling(p)
        # p = th.flatten(p, start_dim=1)
        # p = self.fc(p)
        # p = F.tanh(p)
        p = 0.01*p.view(-1,6)
        return p