import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
from models.resnet_2d import Bottleneck, conv1x1, conv3x3
from models.SE_Resnet import BottleneckX
from models.temporal_xception import TemporalXception


class StNet(nn.Module):
    def __init__(
        self, block, layers, cardinality=32, num_classes=400, T=7, N=5, input_channels=3
    ):
        super(StNet, self).__init__()
        self.inplanes = 64
        self.cardinality = cardinality
        self.T = T
        self.N = N
        self.conv1 = nn.Conv2d(
            input_channels * self.N, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.temp1 = TemporalBlock(256)
        self.temp2 = TemporalBlock(512)
        self.xception = TemporalXception(1024, 2048)
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, self.cardinality, stride, downsample)
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        # size (batch_size, channels, video_length = T * N, height, width)
        B, C, L, H, W = x.size()
        assert self.T * self.N == L
        x = x.view(B * self.T, self.N * C, H, W)
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        # size (batch_size*T, Ci, Hi, Wi)
        size = x.size()
        x = x.view(B, self.T, size[1], size[2], size[3])
        x = self.temp1(x)
        x = self.layer3(x)
        # size (batch_size*T, Ci, Hi, Wi)
        size = x.size()
        x = x.view(B, self.T, size[1], size[2], size[3])
        x = self.temp2(x)
        x = self.layer4(x)
        # size (batch_size*T, Ci, Hi, Wi)
        size = x.size()
        x = F.avg_pool2d(x, kernel_size=(size[2], size[3]))
        # size (batch_size*T, Ci, 1, 1)
        x = x.view(B, size[1], self.T)
        # size (batch_size, T, Ci)
        x = self.xception(x)
        x = self.fc(x)

        return x


class TemporalBlock(nn.Module):
    def __init__(self, channels):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv3d(
            channels,
            channels,
            kernel_size=(3, 1, 1),
            stride=1,
            padding=(1, 0, 0),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x.view(B * T, C, H, W)
        return x


def stnet50(**kwargs):
    """
    Construct stnet with a SE-Resnext 50 backbone.
    """

    model = StNet(BottleneckX, [3, 4, 6, 3], **kwargs)
    return model
