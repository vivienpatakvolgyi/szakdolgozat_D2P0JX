import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

from _classes import Transformer2_3_1


class resnet_lstm(torch.nn.Module):
    def __init__(self):
        super(resnet_lstm, self).__init__()
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.share = torch.nn.Sequential()
        self.share.add_module("conv1", resnet.conv1)
        self.share.add_module("bn1", resnet.bn1)
        self.share.add_module("relu", resnet.relu)
        self.share.add_module("maxpool", resnet.maxpool)
        self.share.add_module("layer1", resnet.layer1)
        self.share.add_module("layer2", resnet.layer2)
        self.share.add_module("layer3", resnet.layer3)
        self.share.add_module("layer4", resnet.layer4)
        self.share.add_module("avgpool", resnet.avgpool)
        # self.lstm = nn.Linear(2048, 7)
        self.fc = nn.Sequential(nn.Linear(2048, 512),
                                nn.ReLU(),
                                nn.Linear(512, 7))
        # self.dropout = nn.Dropout(p=0.2)
        # self.relu = nn.ReLU()

        # init.xavier_normal_(self.lstm.weight)
        # init.xavier_normal_(self.lstm.all_weights[0][1])
        # init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x = x.view(-1, 3, 224, 224)
        x = self.share.forward(x)
        x = x.view(-1, 2048)
        # self.lstm.flatten_parameters()
        # y = self.relu(self.lstm(x))
        # y = y.contiguous().view(-1, 256)
        # y = self.dropout(y)
        y = self.fc(x)
        return y


class resnet_lstm_LFB(torch.nn.Module):
    def __init__(self):
        super(resnet_lstm_LFB, self).__init__()
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.share = torch.nn.Sequential()
        self.share.add_module("conv1", resnet.conv1)
        self.share.add_module("bn1", resnet.bn1)
        self.share.add_module("relu", resnet.relu)
        self.share.add_module("maxpool", resnet.maxpool)
        self.share.add_module("layer1", resnet.layer1)
        self.share.add_module("layer2", resnet.layer2)
        self.share.add_module("layer3", resnet.layer3)
        self.share.add_module("layer4", resnet.layer4)
        self.share.add_module("avgpool", resnet.avgpool)
        # self.lstm = nn.Linear(2048, 7)
        self.fc = nn.Sequential(nn.Linear(2048, 512),
                                nn.ReLU(),
                                nn.Linear(512, 7))
        # self.dropout = nn.Dropout(p=0.2)
        # self.relu = nn.ReLU()

        # init.xavier_normal_(self.lstm.weight)
        # init.xavier_normal_(self.lstm.all_weights[0][1])
        # init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x = x.view(-1, 3, 224, 224)
        x = self.share.forward(x)
        x = x.view(-1, 2048)
        # self.lstm.flatten_parameters()
        # y = self.relu(self.lstm(x))
        # y = y.contiguous().view(-1, 256)
        # y = self.dropout(y)
        # x = self.fc(x)
        return x


class Transformer(nn.Module):
    def __init__(self, mstcn_f_maps, mstcn_f_dim, out_features, len_q):
        super(Transformer, self).__init__()
        self.num_f_maps = mstcn_f_maps  # 32
        self.dim = mstcn_f_dim  # 2048
        self.num_classes = out_features  # 7
        self.len_q = len_q

        self.transformer = Transformer2_3_1(d_model=out_features, d_ff=mstcn_f_maps, d_k=mstcn_f_maps,
                                            d_v=mstcn_f_maps, n_layers=1, n_heads=8, len_q=len_q)
        self.fc = nn.Linear(mstcn_f_dim, out_features, bias=False)

    def forward(self, x, long_feature):
        out_features = x.transpose(1, 2)
        long_feature = out_features
        inputs = []
        for i in range(out_features.size(1)):
            if i < self.len_q - 1:
                input = torch.zeros((1, self.len_q - 1 - i, self.num_classes)).cuda()
                input = torch.cat([input, out_features[:, 0:i + 1]], dim=1)
            else:
                input = out_features[:, i - self.len_q + 1:i + 1]
            inputs.append(input)
        inputs = torch.stack(inputs, dim=0).squeeze(1)
        feas = torch.tanh(self.fc(long_feature).transpose(0, 1))
        output = self.transformer(inputs, feas)
        # output = output.transpose(1,2)
        # output = self.fc(output)
        return output
