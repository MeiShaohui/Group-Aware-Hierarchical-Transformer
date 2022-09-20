"""
paper:
Hyperspectral Image Classification With Deep Feature Fusion Network
Weiwei Song,, Shutao Li, Leyuan Fang, and Ting Lu
IEEE TGRS, 2018
codes:
Original implementation of DFFN is based on Caffe. Code is available at https://github.com/weiweisong415/Demo_DFFN
This is a PyTorch version of DFFN. Code is available at https://github.com/shangsw/HPDM-SPRN/blob/main/models/DFFN.py
"""
import torch
import torch.nn as nn 
import torch.nn.functional as F


class basic_block(nn.Module):
    # basic residual block
    def __init__(self, filters):
        super(basic_block, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, (3,3), padding=1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, (3,3), padding=1)
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, input):
        x1 = F.relu(self.bn1(self.conv1(input)))
        x2 = self.bn2(self.conv2(x1))
        eltwise = F.relu(input + x2)

        return eltwise


class trans_block(nn.Module):
    # transition block between different stage
    def __init__(self, input_channels, filters):
        super(trans_block, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, filters, (1,1), padding=0, stride=1) #stride=2
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(input_channels, filters, (3,3), padding=1, stride=1) #stride=2
        self.bn2 = nn.BatchNorm2d(filters)
        self.conv2_1 = nn.Conv2d(filters, filters, (3,3), padding=1, stride=1)
        self.bn2_1 = nn.BatchNorm2d(filters)

    def forward(self, input):
        x1 = self.bn1(self.conv1(input))
        x2 = F.relu(self.bn2(self.conv2(input)))
        x2_1 = self.bn2_1(self.conv2_1(x2))
        eltwise = F.relu(x1 + x2_1)

        return eltwise


class DFFN(nn.Module):
    def __init__(self, bands, classes, layers_num, filters=16):
        super(DFFN, self).__init__()
        self.conv1 = nn.Conv2d(bands, filters, (3,3), padding=1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.stage1 = self._make_stage(layers_num[0], filters)

        self.trans1 = trans_block(filters, filters*2)
        self.stage2 = self._make_stage(layers_num[1], filters*2)
        
        self.trans2 = trans_block(filters*2, filters*4)
        self.stage3 = self._make_stage(layers_num[2], filters*4)

        self.conv_stage1 = nn.Conv2d(filters, filters*4, (3,3), padding=1, stride=1)    #stride=4
        self.bn_stage1 = nn.BatchNorm2d(filters*4)
        self.conv_stage2 = nn.Conv2d(filters*2, filters*4, (3,3), padding=1, stride=1)  # stride=2
        self.bn_stage2 = nn.BatchNorm2d(filters*4)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(filters*4, classes)

    def _make_stage(self, block_num, filters):
        layers = []
        for i in range(block_num):
            layers.append(basic_block(filters))

        return nn.Sequential(*layers)

    def forward(self, input):
        input = input.squeeze(1)
        # stage 1
        x = F.relu(self.bn1(self.conv1(input)))
        eltwise4 = self.stage1(x)

        # stage 2
        eltwise5 = self.trans1(eltwise4)
        eltwise8 = self.stage2(eltwise5)

        # stage 3
        eltwise9 = self.trans2(eltwise8)
        eltwise12 = self.stage3(eltwise9)
        
        # fusion
        conv_eltwise4 = self.bn_stage1(self.conv_stage1(eltwise4))
        conv_eltwise8 = self.bn_stage2(self.conv_stage2(eltwise8))

        fuse2 = conv_eltwise4 + conv_eltwise8 + eltwise12
        pool = self.avgpool(fuse2)
        out = self.fc(pool.flatten(1))

        return out


def dffn(dataset, patch_size):
    model = None
    if dataset == 'sa':
        model = DFFN(bands=204, classes=16, layers_num=[3,3,3])

    elif dataset == 'pu':
        model = DFFN(bands=103, classes=9, layers_num=[4,4,4])

    elif dataset == 'whulk':
        model = DFFN(bands=270, classes=9, layers_num=[4,4,4]) 

    elif dataset == 'hrl':
        model = DFFN(bands=176, classes=14, layers_num=[4,4,4])

    return model


if __name__ == '__main__':
    t = torch.randn(size=(3, 1, 204, 7, 7))
    print("input shape:", t.shape)
    net = dffn(dataset='sa', patch_size=7)
    print("output shape:", net(t).shape)

