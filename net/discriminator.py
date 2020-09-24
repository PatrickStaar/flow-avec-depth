  
# -*- coding: utf-8 -*-
# @Author: aaronlai

import torch.nn as nn
import torch.nn.functional as F
import torch


class DCGAN_Discriminator(nn.Module):

    def __init__(self, featmap_dim=512, n_channel=1):
        super(DCGAN_Discriminator, self).__init__()
        self.featmap_dim = featmap_dim
        self.conv1 = nn.Conv2d(n_channel, int(featmap_dim / 4), 5,
                               stride=2, padding=2)
        
        self.conv2 = nn.Conv2d(int(featmap_dim / 4), int(featmap_dim / 2), 5,
                               stride=2, padding=2)
        self.BN2 = nn.BatchNorm2d(int(featmap_dim / 2))

        self.conv3 = nn.Conv2d(int(featmap_dim / 2), featmap_dim, 5,
                               stride=2, padding=2)
        self.BN3 = nn.BatchNorm2d(featmap_dim)

        # self.fc = nn.Linear(featmap_dim * 4 * 4, 1)

    def forward(self, x):
        """
        Strided convulation layers,
        Batch Normalization after convulation but not at input layer,
        LeakyReLU activation function with slope 0.2.
        """
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = F.leaky_relu(self.BN2(self.conv2(x)), negative_slope=0.2)
        x = F.leaky_relu(self.BN3(self.conv3(x)), negative_slope=0.2)
        x = torch.sigmoid(x)
        x = x.mean(3).mean(2)
        # x = x.view(-1, self.featmap_dim * 4 * 4)
        # x = F.sigmoid(self.fc(x))
        return x