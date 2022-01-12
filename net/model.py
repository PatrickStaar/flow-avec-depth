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
        init_weights(self.depth_net.modules())
        init_weights(self.pose_net.modules())
    
    def load(self, path):
        load(self.depth_net,path.get('depth'))
        load(self.pose_net,path.get('pose'))


