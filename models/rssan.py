"""
codes:
modified from https://github.com/lierererniu/RSSAN-Hyperspectral-Image
paper:
Residual Spectral-Spatial Attention Network for Hyperspectral Image Classification
Minghao Zhu, Licheng Jiao, Fang Liu, Shuyuan Yang, and Jianing Wang
IEEE TGRS, 2021
"""

import torch
from torch import nn
import torch.nn.functional as F


class SpectralAttention(nn.Module):
    def __init__(self, in_features):
        # input = (bs, in_features, H, W)
        # output = (bs, in_features, H, W)
        super().__init__()
        self.in_features = in_features
        hidden_features = self.in_features // 8  # 8 from original paper
        self.AvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.MaxPool = nn.AdaptiveMaxPool2d((1, 1))
        self.SharedMLP = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, in_features),
            nn.Sigmoid()
        )
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.AvgPool(x)  # z_avg
        x2 = self.MaxPool(x)  # z_max

        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)

        x1 = self.SharedMLP(x1)  # s_avg
        x2 = self.SharedMLP(x2)  # s_max

        s = x1 + x2  # s = s_avg + s_max
        s = s.reshape(s.size(0), self.in_features, 1, 1)
        s = self.Sigmoid(s)  # this step exists in original CBAM, but not mentioned in RSSAN paper
        return s * x  # broadcasting, h'_c = s_c * h_c


class SpatialAttention(nn.Module):
    # input = (bs, in_features, H, W)
    # output = (bs, in_features, H, W)
    def __init__(self):
        super().__init__()
        self.Conv2d = nn.Conv2d(
            in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0
            )
        self.Sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x1 = torch.mean(x, dim=1, keepdim=True)  # global average pooling
        x2, _ = torch.max(x, dim=1, keepdim=True) # global max pooling
        sa = self.Sigmoid(self.Conv2d(torch.cat((x1, x2), dim=1)))
        # (bs, 1, H, W) * (bs, in_features, H, W) = (bs, in_features, H, W)
        return sa * x # broadcasting, h'' = s_a * h'


class SSA(nn.Module):
    # input = (bs, in_features, H, W)
    # output = (bs, in_features, H, W)
    def __init__(self, in_features):
        super().__init__()
        self.SeAM = SpectralAttention(in_features=in_features)
        self.SaAM = SpatialAttention()

    def forward(self, x):
        return(self.SaAM(self.SeAM(x)))


class RSSA(nn.Module):
    # input = (bs, in_features, H, W)
    # output = (bs, in_features, H, W)
    def __init__(self, in_features, kernel_number):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_features, out_channels=kernel_number, kernel_size=3, stride=1, padding=1
            )
        self.conv2 = nn.Conv2d(
            in_channels=kernel_number, out_channels=in_features, kernel_size=3, stride=1, padding=1
            )
        self.bn1 = nn.BatchNorm2d(num_features=kernel_number)
        self.bn2 = nn.BatchNorm2d(num_features=in_features)
        self.ssa = SSA(in_features=in_features)

    def forward(self, x):
        res = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))
        return F.relu((res * self.ssa(res)) + x)

class RSSAN(nn.Module):
    def __init__(self, n_bands, kernel_number, patch_size, n_classes):
        super().__init__()
        self.ssa = SSA(in_features=n_bands)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=n_bands, out_channels=kernel_number, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=kernel_number),
            nn.ReLU(inplace=True)
        )
        self.rssa1 = RSSA(in_features=kernel_number, kernel_number=kernel_number)
        self.rssa2 = RSSA(in_features=kernel_number, kernel_number=kernel_number)
        self.conv1x1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.fc = nn.Linear(patch_size**2, n_classes)

    def forward(self, x):
        x = x.squeeze(dim=1)
        x = self.ssa(x)
        x = self.block1(x)
        x = self.rssa1(x)
        x = self.rssa2(x)
        x = torch.mean(x, dim=1, keepdim=True)
        x = self.conv1x1(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x
    

def rssan(dataset, patch_size):
    model = None
    if dataset == 'sa':
        model = RSSAN(n_bands=204, kernel_number=32, patch_size=patch_size, n_classes=16)
    elif dataset == 'pu':
        model = RSSAN(n_bands=103, kernel_number=32, patch_size=patch_size, n_classes=9)
    elif dataset == 'whulk':
        model = RSSAN(n_bands=270, kernel_number=32, patch_size=patch_size, n_classes=9)
    elif dataset == 'hrl':
        model = RSSAN(n_bands=176, kernel_number=32, patch_size=patch_size, n_classes=14)
    return model


if __name__ == '__main__':
    t = torch.randn(size=(3, 1, 204, 7, 7))
    print("input shape:", t.shape)
    net = rssan(dataset='sa', patch_size=7)
    print("output shape:", net(t).shape)


