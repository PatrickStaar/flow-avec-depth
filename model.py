import torch as th
from torch import nn
import torch.nn.functional as F
import torchvision
from resnet import BasicBlock, Bottleneck
from collections import OrderedDict


## main model

def conv(in_channels, out_channels, k=3, stride=2, padding=1, output=False):
    if output:
        return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, k, stride, padding, bias=False,),
                    nn.Tanh()
                )
    else:
        return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, k, stride, padding, bias=False,),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
                )



def deconv(inchannel, outchannel):
    return nn.Sequential(
        # 由于卷积核滑动过程中，边界情况的不确定，使得在运算步长大于 1 的反卷积时会出现多种合法输出尺寸
        # pytorch 的反卷积层提供了 output_padding 供使用者选择输出，一般情况下我们希望输入输出尺寸以步长为比例
        # 因此 output_padding 一般取 stride-1，同时 padding 取 (kernel_size - 1)/2
        nn.ConvTranspose2d(inchannel, outchannel, kernel_size=3,
                           stride=2, padding=1, output_padding=1, bias=False),
        # nn.BatchNorm2d(cfg[1]),
        nn.ReLU()
    )


def deconv_group(inchannel, addtional):
    return OrderedDict([
            ("layer_{}".format(i+1), deconv(inchannel[i]+addtional[i], inchannel[i+1])) for i in range(5)
        ])


def cat(x):
    return th.cat(x, -3)


class PDF(nn.Module):
    def __init__(self):

        super(PDF, self).__init__()
        
        self.inputs = nn.Conv2d(6, 3, kernel_size=7,
                                stride=2, padding=3, bias=False)

        conv_channels = [64, 64, 128, 256, 512]
        deconv_channels = [512, 256, 128, 64, 32, 16]
        depth_output_channels = [0, 0, 2, 2, 2, 2]
        flow_output_channels = [0, 0, 4, 4, 4, 4]
        block_num = [3, 4, 6, 3]
        self.dilation = 1
        self.groups = 1
        self.base_width = 64

        self.conv1 = conv(3, 32, k=7, stride=2, padding=3)

        # resblocks of 4 size
        blocks = OrderedDict([
            ("layer_{}".format(i+1),
                self._make_layer(
                    conv_channels[i], BasicBlock, conv_channels[i+1], blocks=block_num[i], stride=2)
             ) for i in range(4)
        ])

        self.resblocks = nn.Sequential(blocks)

        self.deconv_depth = nn.Sequential(deconv_group(deconv_channels,depth_output_channels))
        self.deconv_flow = nn.Sequential(deconv_group(deconv_channels,flow_output_channels))

        self.conv1x1_1 = conv(512*2, 512, k=1, stride=1, padding=0)
        self.conv1x1_2 = conv(256*2, 256, k=1, stride=1, padding=0)
        self.conv1x1_3 = conv(128*2, 128, k=1, stride=1, padding=0)
        self.conv1x1_4 = conv(64*2, 64, k=1, stride=1, padding=0)
        self.conv1x1_5 = conv(64+32, 32, k=1, stride=1, padding=0)

        # for depth
        self.another_conv = nn.Sequential(
            conv(512, 512, stride=1),
            conv(512, 512, stride=1),
            conv(512, 512, stride=1),
        )

        self.output2_depth = conv(deconv_channels[2], 2, k=1, stride=1, padding=0,output=True)
        self.output3_depth = conv(deconv_channels[3], 2, k=1, stride=1, padding=0,output=True)
        self.output4_depth = conv(deconv_channels[4], 2, k=1, stride=1, padding=0,output=True)
        self.output5_depth = conv(deconv_channels[5], 2, k=1, stride=1, padding=0,output=True)

        self.output2_flow = conv(deconv_channels[2], 4, k=1, stride=1, padding=0,output=True)
        self.output3_flow = conv(deconv_channels[3], 4, k=1, stride=1, padding=0,output=True)
        self.output4_flow = conv(deconv_channels[4], 4, k=1, stride=1, padding=0,output=True)
        self.output5_flow = conv(deconv_channels[5], 4, k=1, stride=1, padding=0,output=True)

        self.pose_estmation = nn.Sequential(
            conv(512, 512),
            conv(512, 256),
            conv(256, 128),
        )

        self.fc = nn.Sequential(
            nn.Linear(2*3*128, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 6)
        )

    def forward(self, inputs):
        x, y = inputs
        input1 = self.conv1(x)  # 32
        input2 = self.conv1(y)  # 32
        x = cat([input1, input2])  # 64

        x1 = self.resblocks.layer_1(x)
        x2 = self.resblocks.layer_2(x1)
        x3 = self.resblocks.layer_3(x2)
        x4 = self.resblocks.layer_4(x3)

        # feature=F.max_pool2d(kernel_size=2,stride=2)

        # Depth Part
        d = self.another_conv(x4)

        d = cat([d, x4])
        d = self.conv1x1_1(d)
        d = self.deconv_depth.layer_1(d)  # 512->256
        # d1 = self.output1(d1)

        d = cat([d, x3])
        d = self.conv1x1_2(d)
        d = self.deconv_depth.layer_2(d)  # 256->128
        d2 = self.output2_depth(d)

        d = cat([d, x2])
        d = self.conv1x1_3(d)
        d = self.deconv_depth.layer_3(cat([d, d2]))  # 128->64
        d3 = self.output3_depth(d)

        d = cat([d, x1])
        d = self.conv1x1_4(d)
        d = self.deconv_depth.layer_4(cat([d, d3]))  # 64->32
        d4 = self.output4_depth(d)

        d = cat([d, x])
        d = self.conv1x1_5(d)
        d = self.deconv_depth.layer_5(cat([d, d4]))  # 32->16
        d5 = self.output5_depth(d)

        # Pose Part
        p = self.pose_estmation(x4)
        p = th.flatten(p, start_dim=1)
        p = self.fc(p)

        # Flow Part
        f = self.another_conv(x4)

        f = cat([f, x4])
        f = self.conv1x1_1(f)
        f = self.deconv_flow.layer_1(f)  # 512->256
        # f1 = self.output1(f1)

        f = cat([f, x3])
        f = self.conv1x1_2(f)
        f = self.deconv_flow.layer_2(f)  # 256->128
        f2 = self.output2_flow(f)

        f = cat([f, x2])
        f = self.conv1x1_3(f)
        f = self.deconv_flow.layer_3(cat([f, f2]))  # 128->64
        f3 = self.output3_flow(f)

        f = cat([f, x1])
        f = self.conv1x1_4(f)
        f = self.deconv_flow.layer_4(cat([f, f3]))  # 64->32
        f4 = self.output4_flow(f)

        f = cat([f, x])
        f = self.conv1x1_5(f)
        f = self.deconv_flow.layer_5(cat([f, f4]))  # 32->16
        f5 = self.output5_flow(f)

        if self.training:
            return [d5, d4, d3, d2], p, [f5, f4, f3, f2]
        else:
            return [d5,],p,[f5,]

    def init_weights(self):
        # weights train from scratch
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_layer(self, inchannels, block, channels, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or inchannels != channels * block.expansion:
            downsample = nn.Sequential(
                conv(inchannels, channels * block.expansion,
                     stride=stride, k=1, padding=0),
            )

        layers = []
        layers.append(block(inchannels, channels, stride, downsample, 
                self.groups, self.base_width, previous_dilation))

        # resnet34 中暂且不用考虑 expansion
        inchannels = channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(inchannels, channels))

        return nn.Sequential(*layers)
