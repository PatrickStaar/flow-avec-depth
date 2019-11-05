import torch as th
from torch import nn
import torch.nn.functional as F
import torchvision
from resnet import resnet50
from collections import OrderedDict


## ResNet
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(planes),
        nn.ReLU(inplace=True)
    ) 

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
        self.base = features(config['base'])

        conv_planes = [32, 64, 128, 256, 512, 512, 512]
        deconv_planes = [512, 512, 256, 128, 64, 32]

        self.conv1 = conv(3, conv_planes[0], k=7,stride=2,padding=3)


        blocks = OrderedDict([
            ("feature_{}".format(i+2), 
                make_layer(conv_planes[i],BasicBlock, conv_planes[i+1],blocks=2,stride=2)
            ) for i in range(6)
        ]
)
        self.resblocks=nn.Sequential(blocks)

        deconvs=OrderedDict([
            ("deconv_{}".format(i),deconv(deconv_planes[i],deconv_planes[i+1])) for i in range(5)
        ])

        self.deconvs=nn.Sequential(deconvs)

        # self.conv2 = make_layer(conv_planes[0], BasicBlock, conv_planes[1], blocks=2, stride=2)
        # self.conv3 = make_layer(conv_planes[1], BasicBlock, conv_planes[2], blocks=2, stride=2)
        # self.conv4 = make_layer(conv_planes[2], BasicBlock, conv_planes[3], blocks=2, stride=2)
        # self.conv5 = make_layer(conv_planes[3], BasicBlock, conv_planes[4], blocks=2, stride=2)
        # self.conv6 = make_layer(conv_planes[4], BasicBlock, conv_planes[5], blocks=2, stride=2)
        # self.conv7 = make_layer(conv_planes[5], BasicBlock, conv_planes[6], blocks=2, stride=2)

    def forward(self,x):
        x,y=x
        input1=self.conv1(x)
        input2=self.conv1(y)
        x=th.cat([input1,input2],dim=-3)
        x1=self.resblocks.conv_1(x)
        x2=self.resblocks.conv_2(x1)
        x3=self.resblocks.conv_3(x2)
        x4=self.resblocks.conv_4(x3)
        x5=self.resblocks.conv_5(x4)
        x6=self.resblocks.conv_6(x5)

        
        return x

    def init_weights(self):
        # weights train from scratch
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()




class end2end(nn.Module):
    def __init__(self,config):
        super(end2end,self).__init__()
        
        self.features=Features(config['features'])
        self.deconv1=deconv(512,6),
        self.deconv2=deconv(256,6),
        self.deconv3=deconv(128,6),
        self.deconv4=deconv(64,6),

        self.fc=nn.Sequential(
            nn.Linear(7*7*512,512),
            nn.Sigmoid()
            nn.Linear(512,256),
            nn.Sigmoid()
            nn.Linear(256,6)
        )

        deconv_planes = [512, 512, 256, 128, 64, 32, 16]
        self.deconv7 = deconv(conv_planes[6],   deconv_planes[0])
        self.deconv6 = deconv(deconv_planes[0], deconv_planes[1])
        self.deconv5 = deconv(deconv_planes[1], deconv_planes[2])
        self.deconv4 = deconv(deconv_planes[2], deconv_planes[3])
        self.deconv3 = deconv(deconv_planes[3], deconv_planes[4])
        self.deconv2 = deconv(deconv_planes[4], deconv_planes[5])
        self.deconv1 = deconv(deconv_planes[5], deconv_planes[6])

        self.iconv7 = make_layer(deconv_planes[0] + conv_planes[5], BasicBlock, deconv_planes[0], blocks=1, stride=1)
        self.iconv6 = make_layer(deconv_planes[1] + conv_planes[4], BasicBlock, deconv_planes[1], blocks=1, stride=1)
        self.iconv5 = make_layer(deconv_planes[2] + conv_planes[3], BasicBlock, deconv_planes[2], blocks=1, stride=1)
        self.iconv4 = make_layer(deconv_planes[3] + conv_planes[2], BasicBlock, deconv_planes[3], blocks=1, stride=1)
        self.iconv3 = make_layer(1 + deconv_planes[4] + conv_planes[1], BasicBlock, deconv_planes[4], blocks=1, stride=1)
        self.iconv2 = make_layer(1 + deconv_planes[5] + conv_planes[0], BasicBlock, deconv_planes[5], blocks=1, stride=1)
        self.iconv1 = make_layer(1 + deconv_planes[6], BasicBlock, deconv_planes[6], blocks=1, stride=1)

        self.predict_disp6 = conv(deconv_planes[1],1)
        self.predict_disp5 = conv(deconv_planes[2],1)
        self.predict_disp4 = conv(deconv_planes[3],1)
        self.predict_disp3 = conv(deconv_planes[4],1)
        self.predict_disp2 = conv(deconv_planes[5],1)
        self.predict_disp1 = conv(deconv_planes[6],1)



    def forward(self,x):
        x=features(x)
        d=self.deconv1(x)
        d=self.deconv2(x)
        d=self.deconv3(x)
        d=self.deconv4(x)

        f=self.deconv1(x)
        f=self.deconv2(f)
        f=self.deconv3(f)
        f=self.deconv4(f)


        x = th.flatten(x,1)
        p=self.fc(x)
