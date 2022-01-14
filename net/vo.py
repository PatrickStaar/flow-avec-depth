from torch import nn
import torch.nn.functional as F
from .utils import *
from . import resnet50


class Disp(nn.Module):
    def __init__(self):

        super(Disp, self).__init__()
        self.encoder = resnet50(
            input_channels=3, no_top=True, multi_outputs=True)

        conv_channels = [64, 256, 512, 1024, 2048]
        # deconv_channels = [512, 256, 128, 64, 32, 16]
        deconv_channels = [2048, 1024, 512, 256, 64, 32]

        self.decode1 = deconv(deconv_channels[0], deconv_channels[1])
        self.decode2 = deconv(deconv_channels[1], deconv_channels[2])
        self.decode3 = deconv(deconv_channels[2]+1, deconv_channels[3])
        self.decode4 = deconv(deconv_channels[3]+1, deconv_channels[4])
        self.decode5 = deconv(deconv_channels[4]+1, deconv_channels[5])

        self.conv1x1_2 = conv(
            deconv_channels[1]*2, deconv_channels[1], k=1, stride=1, padding=0)
        self.conv1x1_3 = conv(
            deconv_channels[2]*2, deconv_channels[2], k=1, stride=1, padding=0)
        self.conv1x1_4 = conv(
            deconv_channels[3]*2, deconv_channels[3], k=1, stride=1, padding=0)
        self.conv1x1_5 = conv(
            deconv_channels[4]*2, deconv_channels[4], k=1, stride=1, padding=0)

        self.output2 = conv(
            deconv_channels[2], 1, k=1, stride=1, padding=0, output=True)
        self.output3 = conv(
            deconv_channels[3], 1, k=1, stride=1, padding=0, output=True)
        self.output4 = conv(
            deconv_channels[4], 1, k=1, stride=1, padding=0, output=True)
        self.output5 = conv(
            deconv_channels[5], 1, k=1, stride=1, padding=0, output=True)

    def forward(self, inputs):
        x = self.encoder(inputs)  # 7x7

        d = self.decode1(x[4])  # 2048->1024, 14x14

        d = self.conv1x1_2(cat([d, x[3]]))  # 1024+1024->1024
        d = self.decode2(d)  # 1024->512, 28x28
        d1 = self.output2(d)  # 1x28x28
        d1 = F.sigmoid(d1)

        d = self.conv1x1_3(cat([d, x[2]]))  # 512+512->512, 28x28
        d = self.decode3(cat([d, d1]))  # 256x56x56
        d2 = self.output3(d)  # 1x56x56
        d2 = F.sigmoid(d2)

        d = self.conv1x1_4(cat([d, x[1]]))  # 256+256->256, 56x56
        d = self.decode4(cat([d, d2]))  # 128x112x112
        d3 = self.output4(d)  # 1x112x112
        d3 = F.sigmoid(d3)

        d = self.conv1x1_5(cat([d, x[0]]))  # 128+128->128, 112x112
        d = self.decode5(cat([d, d3]))  # 64x224x224
        d4 = self.output5(d)  # 1x224x224
        d4 = F.sigmoid(d4)

        return [d4, d3, d2, d1]

    def init_weights(self):
        init_weights(self.modules())
    
    def load(self, path):
        load(self, path.get('depth'))

class Disp(nn.Module):
    def __init__(self, input_channels=3):

        super(Disp, self).__init__()
        self.encoder = resnet50(
            input_channels=input_channels, no_top=True, multi_outputs=True)

        conv_channels = [64, 256, 512, 1024, 2048]
        # deconv_channels = [512, 256, 128, 64, 32, 16]
        deconv_channels = [2048, 1024, 512, 256, 64, 32]

        self.decode1 = deconv(deconv_channels[0], deconv_channels[1])
        self.decode2 = deconv(deconv_channels[1], deconv_channels[2])
        self.decode3 = deconv(deconv_channels[2]+1, deconv_channels[3])
        self.decode4 = deconv(deconv_channels[3]+1, deconv_channels[4])
        self.decode5 = deconv(deconv_channels[4]+1, deconv_channels[5])

        self.conv1x1_2 = conv(
            deconv_channels[1]*2, deconv_channels[1], k=1, stride=1, padding=0)
        self.conv1x1_3 = conv(
            deconv_channels[2]*2, deconv_channels[2], k=1, stride=1, padding=0)
        self.conv1x1_4 = conv(
            deconv_channels[3]*2, deconv_channels[3], k=1, stride=1, padding=0)
        self.conv1x1_5 = conv(
            deconv_channels[4]*2, deconv_channels[4], k=1, stride=1, padding=0)

        self.output2 = conv(
            deconv_channels[2], 1, k=1, stride=1, padding=0, output=True)
        self.output3 = conv(
            deconv_channels[3], 1, k=1, stride=1, padding=0, output=True)
        self.output4 = conv(
            deconv_channels[4], 1, k=1, stride=1, padding=0, output=True)
        self.output5 = conv(
            deconv_channels[5], 1, k=1, stride=1, padding=0, output=True)

    def forward(self, inputs):
        x = self.encoder(inputs)  # 7x7

        d = self.decode1(x[4])  # 2048->1024, 14x14

        d = self.conv1x1_2(cat([d, x[3]]))  # 1024+1024->1024
        d = self.decode2(d)  # 1024->512, 28x28
        d1 = self.output2(d)  # 1x28x28
        d1 = F.sigmoid(d1)

        d = self.conv1x1_3(cat([d, x[2]]))  # 512+512->512, 28x28
        d = self.decode3(cat([d, d1]))  # 256x56x56
        d2 = self.output3(d)  # 1x56x56
        d2 = F.sigmoid(d2)

        d = self.conv1x1_4(cat([d, x[1]]))  # 256+256->256, 56x56
        d = self.decode4(cat([d, d2]))  # 128x112x112
        d3 = self.output4(d)  # 1x112x112
        d3 = F.sigmoid(d3)

        d = self.conv1x1_5(cat([d, x[0]]))  # 128+128->128, 112x112
        d = self.decode5(cat([d, d3]))  # 64x224x224
        d4 = self.output5(d)  # 1x224x224
        d4 = F.sigmoid(d4)

        return [d4, d3, d2, d1]

    def init_weights(self):
        init_weights(self.modules())
    
    def load(self, path):
        load(self, path.get('depth'))

        
class Pose(nn.Module):
    def __init__(self):

        super(Pose, self).__init__()

        self.conv_block = nn.Sequential(
            conv(2048, 1024, activation='relu'),
            conv(1024, 512, activation='relu'),
            conv(512, 256, activation='relu'),
        )
        self.adaptive_pooling = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(256, 6)
        )

    def forward(self, features):
        p = self.conv_block(features[-1])
        p = p.mean(3).mean(2)
        # p = self.adaptive_pooling(p)
        p = th.flatten(p, start_dim=1)
        p = self.fc(p)
        # p = F.tanh(p)
        return p
