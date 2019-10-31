import torch as th
from torch import nn
import torch.nn.functional as F
import torchvision
from resnet import resnet50
from collections import OrderedDict


def conv(cfg):
    return nn.Sequential(
        nn.Conv2d(cfg[0],cfg[1],cfg[2],cfg[3],padding=cfg[4],bias=False),
        nn.BatchNorm2d(cfg[5]),
        nn.ReLU()
    )


def deconv(cfg):
    return nn.Sequential(
        nn.ConvTranspose2d(cfg[0],cfg[1],cfg[2],cfg[3],padding=cfg[4],bias=False),
        nn.BatchNorm2d(cfg[5]),
        nn.ReLU()
    )


def features(cfg):
    if cfg == 'resnet':
        return resnet50(no_top=True)
    else:
        return nn.Sequential(OrderedDict())


class Features(nn.Module):
    def __init__(self, config):
        super(Features, self).__init__()

        self.inputs = self.conv(config['inputs'])
        self.features = self.features(config['base'])


    def forward(self,x):
        x=self.inputs(x)
        x=self.features(x)
        return x



