import torch as th
from torch import nn
import torch.nn.functional as F
import torchvision
from resnet import resnet50
from collections import OrderedDict


## ResNet
# def conv3x3(in_planes, out_planes, stride=1):
#     return nn.Sequential(
#         nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
#         nn.BatchNorm2d(planes),
#         nn.ReLU(inplace=True)
#     ) 

def conv(in_planes, out_planes, k=3, stride=2, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, k, stride, padding, bias=False,),
        nn.BatchNorm2d(out_planes),
        nn.ReLU()
    )


def deconv(inplane, outplane):
    return nn.Sequential(
        # 由于卷积核滑动过程中，边界情况的不确定，使得在运算步长大于 1 的反卷积时会出现多种合法输出尺寸
        # pytorch 的反卷积层提供了 output_padding 供使用者选择输出，一般情况下我们希望输入输出尺寸以步长为比例
        # 因此 output_padding 一般取 stride-1，同时 padding 取 (kernel_size - 1)/2 
        nn.ConvTranspose2d(inplane,outplane,kernel_size=3,stride=2,padding=1,output_padding=1, bias=false),
        # nn.BatchNorm2d(cfg[1]),
        nn.ReLU()
    )

def cat(x):
    return th.cat(x,-3)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv(inplanes, planes, stride)
        self.conv2 = conv(planes, planes, stride=1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out)
        return out


def make_layer(inplanes, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes))

    return nn.Sequential(*layers)

## End of the Resblock


## main model



# def features(cfg):
#     if cfg == 'resnet':
#         return resnet50(no_top=True)
#     else:
#         return nn.Sequential(OrderedDict())

class Features(nn.Module):
    def __init__(self, config):

        super(Features, self).__init__()
        self.inplane=inplane
        self.inputs = nn.Conv2d(6, 3,kernel_size=7,stride=2,padding=3,bias=False)

        conv_planes = [32, 64, 128, 256, 512, 512, 512]
        deconv_planes = [512, 512, 256, 128, 64, 32]

        self.conv1 = conv(3, conv_planes[0], k=7,stride=2,padding=3)

        blocks = OrderedDict([
            ("feature_{}".format(i+2), 
                make_layer(conv_planes[i],BasicBlock, conv_planes[i+1],blocks=2,stride=2)
            ) for i in range(6)
        ])

        self.resblocks=nn.Sequential(blocks)

        deconvs=OrderedDict([
            ("deconv_{}".format(i+1),deconv(deconv_planes[i],deconv_planes[i+1])) for i in range(5)
        ])

        self.deconvs=nn.Sequential(deconvs)

        self.output1=deconv(deconv_planes[1]*2,1)
        self.output2=deconv(deconv_planes[2]*2,1)
        self.output3=deconv(deconv_planes[3]*2,1)
        self.output4=deconv(deconv_planes[4]*2,1)
        self.output5=deconv(deconv_planes[5]*2,1)
        
        self.pose_estmation=nn.Sequential(
            conv(512, 512),
            conv(512, 256),
            conv(256, 64),
        )

        self.fc=nn.Sequential(
            nn.Linear(7*7*512,512),
            nn.Sigmoid(),
            nn.Linear(512,256),
            nn.Sigmoid(),
            nn.Linear(256,6)
        )


    def forward(self,x):
        x,y=x
        input1=self.conv1(x)
        input2=self.conv1(y)
        x=cat([input1,input2])

        x1=self.resblocks.conv_1(x)
        x2=self.resblocks.conv_2(x1)
        x3=self.resblocks.conv_3(x2)
        x4=self.resblocks.conv_4(x3)
        x5=self.resblocks.conv_5(x4)
        x6=self.resblocks.conv_6(x5)

        feature=F.max_pool2d(kernel_size=2,stride=2)

        # Depth Part
        d1=self.deconvs.deconv_1(feature)
        d2=self.deconvs.deconv_2(cat([d1,x4]))
        d3=self.deconvs.deconv_3(cat([d1,x3]))
        d4=self.deconvs.deconv_4(cat([d1,x2]))
        d5=self.deconvs.deconv_5(cat([d1,x1]))

        d_1_out = self.output1(d1)
        d_2_out = self.output1(d2)
        d_3_out = self.output1(d3)
        d_4_out = self.output1(d4)
        d_5_out = self.output1(d5)
       
        # Pose Part
        p = self.pose_estmation(feature)
        p = th.flatten(p,start_dim=1)
        p = self.fc()

        # Flow Part
        f_1=self.deconvs.deconv_1(feature)
        f_1_out = self.output1(f1)
        f_2=self.deconvs.deconv_2(cat([d1,x4,f_1_out]))
        f_2_out = self.output1(f2)
        f_3=self.deconvs.deconv_3(cat([d1,x3,f_1_out]))
        f_3_out = self.output1(f3)
        f_4=self.deconvs.deconv_4(cat([d1,x2,f_1_out]))
        f_4_out = self.output1(f4)
        f_5=self.deconvs.deconv_5(cat([d1,x1,f_1_out]))
        f_5_out = self.output1(f5)

        return d_5_out,p,f_5_out

    def init_weights(self):
        # weights train from scratch
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
