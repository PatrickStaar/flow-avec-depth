from torch import nn
import torch.nn.functional as F
import torchvision
from . import resnet50
from collections import OrderedDict
from .utils import *


class Flow(nn.Module):
    def __init__(self):

        super(Flow, self).__init__()
        
        conv_channels = [64, 64, 128, 256, 512]
        deconv_channels = [512, 256, 128, 64, 32, 16]

        self.decode1=deconv(deconv_channels[0],deconv_channels[1])
        self.decode2=deconv(deconv_channels[1]+2,deconv_channels[2])
        self.decode3=deconv(deconv_channels[2]+2,deconv_channels[3])
        self.decode4=deconv(deconv_channels[3]+2,deconv_channels[4])
        self.decode5=deconv(deconv_channels[4]+2,deconv_channels[5])

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
        
        f = self.conv_block(features[-1])

        f = self.conv1x1_1(cat([f, features[4]]))
        f = self.decode1(f)  # 512->256
        f1 = self.output1(f)

        f = self.conv1x1_2(cat([f, features[3]]))
        f = self.decode2(cat([f, f1]))  # 256->128
        f2 = self.output2(f)
        f2 = F.relu(f2)

        f = self.conv1x1_3(cat([f, features[2]]))
        f = self.decode3(cat([f, f2]))  # 128->64
        f3 = self.output3(f)
        f3 = F.relu(f3)

        f = self.conv1x1_4(cat([f, features[1]]))
        f = self.decode4(cat([f, f3]))  # 64->32
        f4 = self.output4(f)
        f4 = F.relu(f4)

        f = self.conv1x1_5(cat([f, features[0]]))
        f = self.decode5(cat([f, f4]))  # 32->16
        f5 = self.output5(f)
        f5 = F.relu(f5)

        return [f5, f4, f3, f2, f1]
