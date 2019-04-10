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
        for m in [self.bn1, self.bn2, self.bn3, self.conv]:
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv1d):
                nn.init.dirac_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        B, C, T = x.size()
        x = self.bn1(x)
        x2 = self.conv(x)
        x1 = self.sepconv1(x)
        x1 = F.relu(self.bn2(x1))
        x1 = self.sepconv2(x1)
        # size (B, C, T)
        x = F.relu(self.bn3(x1 + x2)).div(2.0)
        x = F.max_pool1d(x, kernel_size=x.size(-1))
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
        ## init seprabale conv as init
        self.conv1.weight.data.zero_()
        self.conv1.weight[:, :, kernel_size // 2].data.fill_(1)

        self.pointwise = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=bias,
        )
        nn.init.dirac_(self.pointwise.weight)

        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv1d):
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x
