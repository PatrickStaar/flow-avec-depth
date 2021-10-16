import torch
from torch import nn
import torch.nn.functional as F
from .hrnet.lib.models.seg_hrnet import get_seg_model
from .resnet import resnet18
from .utils import *

class FlowBranch(nn.Module):
    def __init__(self, output_channels=512, mode='plain'):
        super(FlowBranch,self).__init__()
        #strides=[2,1,1,1]
        self.layers=resnet18(no_top=True,input_channels=3) 
        #self.layers=resnet34(no_top=True,input_channels=3)
        self.last=nn.Sequential(
            nn.Conv2d(512,output_channels,3,1,1),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(inplace=True),
        )
        # self.init_weights()

    def forward(self, x):
        B,C,H,W=x.shape
        x=torch.cat([x,torch.ones((B,1,H,W)).cuda()],dim=1)
        x=self.layers(x)
        x=self.last(x)
        x=F.interpolate(x,scale_factor=8,mode='bilinear')
        return x

    def init_weights(self):
        for m in self.modules():
            if  isinstance(m, nn.ConvTranspose2d) or \
                isinstance(m,nn.Conv2d) or \
                isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        encoder_dict=torch.load('pretrain/resnet18.pth')
        #encoder_dict=torch.load('pretrain/resnet34.pth')
        self.layers.load_state_dict(encoder_dict,strict=False)


class EncoderHrnet(nn.Module):
    def __init__(self, cfg,fusion=None, flow_branch_grad=False):
        super(EncoderHrnet,self).__init__()
        if fusion is not None:
            self.flow_branch_grad=flow_branch_grad
            self.flow_branch=FlowBranch(720)
            if fusion=='conv':
                self.fusion_layer=nn.Sequential(
                    nn.Conv2d(1440,720,3,1,1),
                    nn.LeakyReLU(inplace=True)
                )

        self.encoder = get_seg_model(cfg,input_channels=3, skip_last_layer=True)
        self.fusion=fusion

    def forward(self, x):
        img,flow=x
        feature=self.encoder(img)
        if self.fusion is not None:
            with torch.set_grad_enabled(self.flow_branch_grad and self.training):
                feature_flow=self.flow_branch(flow)
            if self.fusion=='product':
                feature=feature*feature_flow
            elif self.fusion=='add':
                feature=feature+feature_flow
            else:
                feature=self.fusion_layer(cat([feature,feature_flow]))
            return [feature, feature_flow]

        #features=self.last_layer(features)
        return [feature, None]
    

class DecoderHrnet(nn.Module):
    def __init__(self,cfg, input_channels=720, output_shape=None):

        super(DecoderHrnet,self).__init__()
        BN_MOMENTUM=0.01
        extra=cfg.MODEL.EXTRA
        self.conv_layer1= nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=360,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(360, momentum=BN_MOMENTUM),
            nn.LeakyReLU(inplace=True),
            #nn.Conv2d(
            #    in_channels=input_channels,
            #    out_channels=1,
            #    kernel_size=extra.FINAL_CONV_KERNEL,
            #    stride=1,
            #    padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0)
        )
        self.conv_layer2= nn.Sequential(
            nn.Conv2d(
                in_channels=360,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
            nn.LeakyReLU(inplace=True),
        )
        self.output_layer=nn.Conv2d(
            in_channels=256,
            out_channels=1,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0)

 
        self.output_shape=output_shape
        # self.init_weights()

    def forward(self, x):
        x=self.conv_layer1(x)
        x=F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False)
        x=self.conv_layer2(x)
        x=F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False)
        x=self.output_layer(x)

        #x=self.layers(x)
        if self.output_shape is not None:
            x=F.interpolate(x,self.output_shape,mode='bilinear',align_corners=False)
        return x
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)