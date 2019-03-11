import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial


class TemporalXception(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TemporalXception, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.sepconv1 = SeparableConv1d(
            in_channels, out_channels, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.sepconv2 = SeparableConv1d(
            out_channels, out_channels, kernel_size=3, padding=1
        )
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        B, C, T = x.size()
        # print('xception input', x.size())
        x = self.bn1(x)
        x2 = self.conv(x)
        # print('xception conv', x2.size())
        x1 = self.sepconv1(x)
        # print('xception sepconv1', x1.size())
        x1 = F.relu(self.bn2(x1))
        # print('xception bn1', x1.size())
        x1 = self.sepconv2(x1)
        # print('xception sepconv2', x1.size())
        # size (B, T, C)
        x = F.relu(self.bn3(x1 + x2))
        # print('xception bn3', x1.size())
        x = F.max_pool1d(x, kernel_size=x.size(-1))
        # print('xception maxpool', x1.size())
        # size (B,C,1)
        return x.view(x.size(0), x.size(1))


class SeparableConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
    ):
        super(SeparableConv1d, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv1d(
            in_channels, out_channels, 1, 1, 0, 1, groups=1, bias=bias
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x
