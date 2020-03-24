import torch as th
from torch import nn
import torch.nn.functional as F
import torchvision
from collections import OrderedDict
from . import Pose, Depth, Flow, resnet50
from .utils import *


## main model
class PDF(nn.Module):
    def __init__(self,**kwargs):
        super(PDF, self).__init__()
        self.use_depth=use_depth
        self.use_flow=use_flow
        self.use_pose=use_pose
        self.encoder = resnet50(input_channels=6,)
        
        if self.use_depth:
            self.depth_net = Depth()
        if self.use_flow:
            self.flow_net = Flow()
        if self.use_pose:
            self.pose_net = Pose()

    def forward(self, inputs):
        x = cat(inputs)
        depth_map=None
        flowmap=None
        pose=None
        features=self.encoder(x)
        if self.use_depth:
            depth_map = self.depth_net(features)
        if self.use_flow:
            flow_map = self.flow_net(features)
        if self.use_pose:
            pose = self.pose_net([features[-1]])

        if self.training:
            return depth_map, pose, flow_map
        else:
            return depth_map[0], pose, flow_map[-1]

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d)\
            or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        #         nn.init.kaiming_uniform(m.weight.data)
        #         if m.bias is not None:
        #             m.bias.data.zero_()
