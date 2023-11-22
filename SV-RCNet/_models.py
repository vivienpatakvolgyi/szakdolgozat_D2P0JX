import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import optim
from torchvision import models
from torchvision.models import ResNet50_Weights


class resnet_lstm(torch.nn.Module):
    def __init__(self):
        super(resnet_lstm, self).__init__()
        net = models.resnet50(weights=ResNet50_Weights.DEFAULT)  # resnet = models.resnet50(pretrained=True)
        self.share = torch.nn.Sequential()
        self.share.add_module("conv1", net.conv1)
        self.share.add_module("bn1", net.bn1)
        self.share.add_module("relu", net.relu)
        self.share.add_module("maxpool", net.maxpool)
        self.share.add_module("layer1", net.layer1)
        self.share.add_module("layer2", net.layer2)
        self.share.add_module("layer3", net.layer3)
        self.share.add_module("layer4", net.layer4)
        self.share.add_module("avgpool", net.avgpool)
        self.lstm = nn.LSTM(2048, 512, batch_first=True)
        self.fc = nn.Linear(512, 7)

        init.xavier_normal(self.lstm.all_weights[0][0])
        init.xavier_normal(self.lstm.all_weights[0][1])
        init.xavier_uniform(self.fc.weight)

    def forward(self, x, sequence_length):
        x = self.share.forward(x)
        x = x.view(-1, 2048)
        x = x.view(-1, sequence_length, 2048)
        self.lstm.flatten_parameters()
        y, _ = self.lstm(x)
        y = y.contiguous().view(-1, 512)
        y = self.fc(y)
        return y

