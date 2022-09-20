"""
codes: github.com/nshaud/DeepHyperx
paper:
3-D Deep Learning Approach for Remote Sensing Image Classification,
Amina Ben Hamida, Alexandre Benoit, Patrick Lambert, Chokri Ben Amar, 
IEEE TGRS, 2018, https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8344565
"""
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import init


class CNN3D(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=7):
        super().__init__()

        # The first layer is a (3,3,3) kernel sized Conv characterized
        # by a stride equal to 1 and number of neurons equal to 20

        self.patch_size = patch_size
        self.input_channels = input_channels

        dilation = 1
        dilation = (dilation, 1, 1)

        if patch_size == 3:
            self.conv1 = nn.Conv3d(
                1, 20, (3, 3, 3), stride=(1, 1, 1), dilation=dilation, padding=1
            )

        else:
            self.conv1 = nn.Conv3d(
                1, 20, (3, 3, 3), stride=(1, 1, 1), dilation=dilation, padding=0
            )

        # Next pooling is applied using a layer identical to the previous one
        # with the difference of a 1D kernel size (1,1,3) and a larger stride
        # equal to 2 in order to reduce the spectral dimension

        self.pool1 = nn.Conv3d(
            20, 20, (3, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0)
        )

        # Then, a duplicate of the first and second layers is created with
        # 35 hidden neurons per layer

        self.conv2 = nn.Conv3d(
            20, 35, (3, 3, 3), dilation=dilation, stride=(1, 1, 1), padding=(1, 0, 0)
        )
        self.pool2 = nn.Conv3d(
            35, 35, (3, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0)
        )

        # Finally, the 1D spatial dimension is progressively reduced
        # thanks to the use of two Conv layers, 35 neurons each,
        # with respective kernel sizes of (1, 1, 3) and (1, 1, 2) and strides
        # respectively equal to (1, 1, 1) and (1, 1, 2)

        self.conv3 = nn.Conv3d(
            35, 35, (3, 1, 1), dilation=dilation, stride=(1, 1, 1), padding=(1, 0, 0)
        )
        self.conv4 = nn.Conv3d(
            35, 35, (2, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0)
        )

        self.dropout = nn.Dropout(p=0.5)

        self.features_size = self._get_final_flattened_size()

        # The architecture ends with a fully connected layer where the number
        # of neurons is equal to the number of input classes

        self.fc = nn.Linear(self.features_size, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, 1, self.input_channels, self.patch_size, self.patch_size)
            )
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.conv3(x)
            x = self.conv4(x)
            s0, s1, s2, s3, s4 = x.size()
        return s1 * s2 * s3 * s4

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, self.features_size)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def cnn3d(dataset, patch_size):
    model = None
    if dataset == 'sa':
        model = CNN3D(input_channels=204, n_classes=16, patch_size=patch_size)
    elif dataset == 'pu':
        model = CNN3D(input_channels=103, n_classes=9, patch_size=patch_size)
    elif dataset == 'whulk':
        model = CNN3D(input_channels=270, n_classes=9, patch_size=patch_size)
    elif dataset == 'hrl':
        model = CNN3D(input_channels=176, n_classes=14, patch_size=patch_size)
    return model


if __name__ == '__main__':
    t = torch.randn(size=(3, 1, 204, 7, 7))
    print("input shape:", t.shape)
    net = cnn3d(dataset='sa', patch_size=7)
    print("output shape:", net(t).shape)


