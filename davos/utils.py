import torch
import cv2
import numpy as np
from torch import nn
from torch.nn.functional import interpolate
import torch.nn.functional as F
from .pwc import pwc_dc_net


def conv(in_channels, out_channels, k=3, stride=2, padding=1, 
        output=False, activation='leaky', bn=False):
    layer=nn.Conv2d(in_channels, out_channels, k, stride, padding, bias=False,)
    if output:
        return layer    
    else:
        act = nn.LeakyReLU(0.1,inplace=True) if activation=='leaky' else nn.ReLU()
        return nn.Sequential(layer, act) # without nn.BatchNorm2d(out_channels)
                        

def deconv(in_channels, out_channels):
    return nn.Sequential(
        # 由于卷积核滑动过程中，边界情况的不确定，使得在运算步长大于 1 的反卷积时会出现多种合法输出尺寸
        # pytorch 的反卷积层提供了 output_padding 供使用者选择输出，一般情况下我们希望输入输出尺寸以步长为比例
        # 因此 output_padding 一般取 stride-1，同时 padding 取 (kernel_size - 1)/2
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3,
                           stride=2, padding=1, output_padding=1, bias=False),
        # without nn.BatchNorm2d(cfg[1]),
        nn.LeakyReLU(0.1,inplace=True) #nn.ReLU()
    )


def cat(x):
    return torch.cat(x, -3)


def build_branch(output_channels=512, shrunk=32):
    strides=[2,2,2,2]
    if shrunk ==8:
        strides[2],strides[3]=1,1
    flow_branch=nn.Sequential(
        nn.Conv2d(2,64,5,strides[0],2,bias=False),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(),
        nn.Conv2d(64,128,3,strides[1],1,bias=False),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(),
        nn.Conv2d(128,256,3,strides[2],1,bias=False),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(),
        nn.Conv2d(256,output_channels,3,strides[3],1,bias=False),
        nn.BatchNorm2d(output_channels),
        nn.LeakyReLU(),
        nn.MaxPool2d(3,2,1)
    )
    return flow_branch


def get_pwc():
    PWC=pwc_dc_net('/work/pretrain/pwc_net.pth').cuda()
    freeze(PWC)
    PWC.eval()
    return PWC


def freeze(net):
    for param in net.parameters():
        param.requires_grad=False


def normalize(tensor):
    B,C,H,W=tensor.size()
    mean=tensor.view(B,C,-1).mean(-1,keepdim=True).unsqueeze(-1)
    if torch.isnan(mean.mean()):
        print('mean')
    std=tensor.view(B,C,-1).std(-1,keepdim=True).unsqueeze(-1)
    if torch.isnan(std.mean()):
        print('std')

    tensor-=mean
    if torch.isnan(tensor.mean()):
        print('sub')

    tensor/=std
    if torch.isnan(tensor.mean()):
        print(std)
        print('divide')

    return tensor


def interp(tensor,size=None,scale=None):
    if size is not None:
        return interpolate(tensor,size=size,mode='bilinear')
    else:
        return interpolate(tensor,scale_factor=scale,mode='bilinear')


def flow_visualize(flow):
    mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
    # flow_norm = (flow[:, :, 0]**2+flow[:, :, 1]**2)**0.5
    # flow_dire = np.arctan2(flow[:, :, 0], flow[:, :, 1])
    # max_value = np.abs(flow[:, :, :2]).max()

    channel0 = ang*180/np.pi
    channel1 = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)
    channel2 = flow[:, :, 2]
    # channel1 = channel1.clip(0,1)
    colormap = np.stack([channel0, channel1, channel2], axis=-1)
    colormap = cv2.cvtColor(np.float32(colormap), cv2.COLOR_HSV2RGB)
    return colormap


def to_numpy(img):
    img = img.detach_().cpu().numpy().squeeze(0).transpose(1, 2, 0)
    img = cv2.cvtColor(img*255, cv2.COLOR_BGR2RGB).astype(np.uint8)
    return img

