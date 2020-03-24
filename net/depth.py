from torch import nn
import torch.nn.functional as F
import torchvision
from collections import OrderedDict
from . import resnet50
from .utils import *


class Depth(nn.Module):
    def __init__(self):

        super(Depth, self).__init__()
        
        conv_channels = [64, 64, 128, 256, 512]
        deconv_channels = [512, 256, 128, 64, 32, 16]

        self.decode1=deconv(deconv_channels[0],deconv_channels[1])
        self.decode2=deconv(deconv_channels[1],deconv_channels[2])
        self.decode3=deconv(deconv_channels[2],deconv_channels[3])
        self.decode4=deconv(deconv_channels[3],deconv_channels[4])
        self.decode5=deconv(deconv_channels[4],deconv_channels[5])

        self.conv1x1_1 = conv(512*2, 512, k=1, stride=1, padding=0)
        self.conv1x1_2 = conv(256*2, 256, k=1, stride=1, padding=0)
        self.conv1x1_3 = conv(128*2, 128, k=1, stride=1, padding=0)
        self.conv1x1_4 = conv(64*2, 64, k=1, stride=1, padding=0)
        self.conv1x1_5 = conv(64+32, 32, k=1, stride=1, padding=0)

        # for depth
        self.conv_block = nn.Sequential(
            conv(512, 512, stride=1,activation='relu'),
            conv(512, 512, stride=1,activation='relu'),
        )

        self.output1 = conv(deconv_channels[1], 2, k=1, stride=1, padding=0,output=True,activation='relu')
        self.output2 = conv(deconv_channels[2], 2, k=1, stride=1, padding=0,output=True,activation='relu')
        self.output3 = conv(deconv_channels[3], 2, k=1, stride=1, padding=0,output=True,activation='relu')
        self.output4 = conv(deconv_channels[4], 2, k=1, stride=1, padding=0,output=True,activation='relu')
        self.output5 = conv(deconv_channels[5], 2, k=1, stride=1, padding=0,output=True,activation='relu')

    def forward(self, features):
        
        d = self.conv_block(features[-1])

        d = self.conv1x1_1(cat([d, features[4]]))
        d = self.decode1(d)  # 512->256
        d1 = self.output1(d1)

        d = self.conv1x1_2(cat([d, features[3]]))
        d = self.decode2(d)  # 256->128
        d2 = self.output2_depth(d)
        d2 = F.relu(d2)

        d = self.conv1x1_3(cat([d, features[2]]))
        d = self.decode3(cat([d, d2]))  # 128->64
        d3 = self.output3_depth(d)
        d3 = F.relu(d3)

        d = self.conv1x1_4(cat([d, features[1]]))
        d = self.decode4(cat([d, d3]))  # 64->32
        d4 = self.output4_depth(d)
        d4 = F.relu(d4)

        d = self.conv1x1_5(cat([d, features[0]]))
        d = self.decode5(cat([d, d4]))  # 32->16
        d5 = self.output5_depth(d)
        d5 = F.relu(d5)

        return [d5, d4, d3, d2, d1]
