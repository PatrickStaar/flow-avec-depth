from torch import nn
import torch.nn.functional as F
import torchvision
from . import resnet50
from collections import OrderedDict
from .utils import *


class Flow(nn.Module):
    def __init__(self):

        super(Flow, self).__init__()
        
        conv_channels = [64, 256, 512, 1024, 2048]
        # deconv_channels = [512, 256, 128, 64, 32, 16]
        deconv_channels = [2048, 1024, 512, 256, 64, 32]

        self.decode1=deconv(deconv_channels[0],deconv_channels[1])
        self.decode2=deconv(deconv_channels[1]+2,deconv_channels[2])
        self.decode3=deconv(deconv_channels[2]+2,deconv_channels[3])
        self.decode4=deconv(deconv_channels[3]+2,deconv_channels[4])
        self.decode5=deconv(deconv_channels[4]+2,deconv_channels[5])

        self.conv1x1_1 = conv(deconv_channels[0]*2, deconv_channels[0], k=1, stride=1, padding=0)
        self.conv1x1_2 = conv(deconv_channels[1]*2, deconv_channels[1], k=1, stride=1, padding=0)
        self.conv1x1_3 = conv(deconv_channels[2]*2, deconv_channels[2], k=1, stride=1, padding=0)
        self.conv1x1_4 = conv(deconv_channels[3]*2, deconv_channels[3], k=1, stride=1, padding=0)
        self.conv1x1_5 = conv(deconv_channels[4]*2, deconv_channels[4], k=1, stride=1, padding=0)

        # for depth
        self.conv_block = nn.Sequential(
            conv(2048, 2048, stride=1,activation='relu'),
            # conv(512, 512, stride=1,activation='relu'),
        )

        self.output1 = conv(deconv_channels[1], 2, k=1, stride=1, padding=0,output=True,activation='relu')
        self.output2 = conv(deconv_channels[2], 2, k=1, stride=1, padding=0,output=True,activation='relu')
        self.output3 = conv(deconv_channels[3], 2, k=1, stride=1, padding=0,output=True,activation='relu')
        self.output4 = conv(deconv_channels[4], 2, k=1, stride=1, padding=0,output=True,activation='relu')
        self.output5 = conv(deconv_channels[5], 2, k=1, stride=1, padding=0,output=True,activation='relu')

    def forward(self, features):
        
        f = self.conv_block(features[-1])

        f = self.conv1x1_1(cat([f, features[4]]))
        f = self.decode1(f)  # 2048->1024
        f1 = self.output1(f)

        f = self.conv1x1_2(cat([f, features[3]]))
        f = self.decode2(cat([f, f1]))  # 1024->512
        f2 = self.output2(f)
        #f2 = F.relu(f2)

        f = self.conv1x1_3(cat([f, features[2]]))
        f = self.decode3(cat([f, f2]))  # 512->256
        f3 = self.output3(f)
        #f3 = F.relu(f3)

        f = self.conv1x1_4(cat([f, features[1]]))
        f = self.decode4(cat([f, f3]))  # 256->64
        f4 = self.output4(f)
        #f4 = F.relu(f4)

        f = self.conv1x1_5(cat([f, features[0]]))
        f = self.decode5(cat([f, f4]))  # 64->32
        f5 = self.output5(f)
        #f5 = F.relu(f5)

        return [f5, f4, f3, f2, f1]
