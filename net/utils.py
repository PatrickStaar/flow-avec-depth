import torch as th
from torch import nn


def conv(in_channels, out_channels, k=3, stride=2, padding=1, 
        output=False, activation='leaky', bn=False):
    layers=[]
    layers.append(nn.Conv2d(in_channels, out_channels, k, stride, padding, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if not output:
        layers.append(nn.LeakyReLU(0.1,inplace=True) if activation=='leaky' else nn.ReLU())
    return nn.Sequential(*layers)
                        

def deconv(in_channels, out_channels):
    return nn.Sequential(
        # 由于卷积核滑动过程中，边界情况的不确定，使得在运算步长大于 1 的反卷积时会出现多种合法输出尺寸
        # pytorch 的反卷积层提供了 output_padding 供使用者选择输出，一般情况下我们希望输入输出尺寸以步长为比例
        # 因此 output_padding 一般取 stride-1，同时 padding 取 (kernel_size - 1)/2
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3,
                           stride=2, padding=1, output_padding=1, bias=False),
        # nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.1,inplace=True) #nn.ReLU()
    )


def cat(x):
    return th.cat(x, -3)