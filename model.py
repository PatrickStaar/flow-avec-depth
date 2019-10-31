import torch as th
from torch import nn
import torchvision
from resnet import resnet50


class Features(nn.Module):
    def __init__(self, config):
        super(Features, self).__init__()

        self.resnet = resnet50(no_top=True)
        self.inputs = nn.Conv2d()

    def conv(self, cfg):
        pass

    def deconv(self, cfg):
        pass
