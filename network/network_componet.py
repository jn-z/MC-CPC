import torch.nn as nn 
import torch.nn.init as init
import torch.nn.functional as F
import torch
import math
import functools
import numpy as np
from collections import OrderedDict
from torch.nn import Parameter
import pdb
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, layers=[3, 4, 6, 3], flatten_dim=4096, spkVec_dim=256):
        self.feature_maps = [16, 16, 32, 64, 128]
        self.inplanes = 16
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, self.feature_maps[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.feature_maps[0])
        self.relu1 = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(self.feature_maps[1], layers[0], stride=1)
        self.layer2 = self._make_layer(self.feature_maps[2], layers[1], stride=2)
        self.layer3 = self._make_layer(self.feature_maps[3], layers[2], stride=2)
        self.layer4 = self._make_layer(self.feature_maps[4], layers[3], stride=2)
        self.fc = nn.Linear(flatten_dim, spkVec_dim)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(spkVec_dim)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * BasicBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * BasicBlock.expansion),
            )

        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * BasicBlock.expansion


        for i in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1, x.size(1), x.size(2))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x_1 = self.layer1(x)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)
        mean_x_4 = torch.mean(x_4,dim=2)
        var_x_4 = torch.var(x_4,dim=2)
        x_5 = torch.cat((mean_x_4,torch.sqrt(var_x_4+0.00001)),dim=2)
        x_5 = x_5.contiguous().view(x.size(0), -1)
        x_6 = self.fc(x_5)
        x_6 = self.relu2(x_6)
        Vector = self.bn2(x_6)

        return Vector

class fullyConnect(nn.Module):
    def __init__(self, target_num=10000,spkVec_dim=256):
        super(fullyConnect, self).__init__()
        self.spkVec_dim = spkVec_dim
        self.target_num = target_num
        self.layer1 = nn.Sequential()
        self.layer1.add_module('linear1', nn.Linear(self.spkVec_dim, self.target_num))
        self.initial_parameters()

    def initial_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0)

    def forward(self, x):
        hiddenVec = self.layer1(x)
        tar = F.softmax(hiddenVec, dim=1)
        return tar
