import torch as th
from torch import nn
import torch.nn.functional as F
import torchvision
from resnet import resnet50
from collections import OrderedDict


def conv(cfg):
    return nn.Sequential(
        nn.Conv2d(cfg[0],cfg[1],cfg[2],cfg[3],padding=cfg[4],bias=False,), # cfg:[in_ch,out_ch,kernel,stride,padding]
        nn.BatchNorm2d(cfg[1]),
        nn.ReLU()
    )


def deconv(inplane, outplane):
    return nn.Sequential(
        nn.ConvTranspose2d(inplane,outplane,kernel_size=3,stride=2,padding=1,bias=false),
        # nn.BatchNorm2d(cfg[1]),
        nn.ReLU()
    )


def features(cfg):
    if cfg == 'resnet':
        return resnet50(no_top=True)
    else:
        return nn.Sequential(OrderedDict())


class Features(nn.Module):
    def __init__(self, config):

        super(Features, self).__init__()
        self.inplane=inplane
        self.inputs = nn.Conv2d(6, 3,kernel_size=7,stride=2,padding=3,bias=False)
        self.base = features(config['base'])


    def forward(self,x):
        x=self.inputs(x)
        x=self.features(x)
        return x

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
