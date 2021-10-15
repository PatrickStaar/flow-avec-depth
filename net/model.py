from time import sleep
import torch as th
from torch import nn
from torch.nn.modules.module import Module
from . import Pose, Depth
from .utils import *


## main model
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.depth_net = Depth()
        self.pose_net = Pose()

    def forward(self, inputs):
        depth_map = self.depth_net(inputs[1])
        pose = self.pose_net(cat(inputs))
        return depth_map if self.training else depth_map[0], pose

    def init_weights(self):
        _init_weights(self.depth_net.modules())
        _init_weights(self.pose_net.modules())
    
    def load(self, path):
        _load(self.depth_net,path.get('depth'))
        _load(self.pose_net,path.get('pose'))


def _init_weights(modules):
    for m in modules:
        if  isinstance(m, nn.Conv2d) or \
            isinstance(m, nn.ConvTranspose2d) or \
            isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def _load(modules, path):
    if path is not None:
        modules.load_state_dict(th.load(path),strict=False)