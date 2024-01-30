# -*- coding:utf -8 -*-
"""
Created on Fri Jan. 12 11:14:11 2024
@author: PJS

** Parameters **
f1 (int): The filter number of block 1. (default: 8)
f2 (int): The filter number of block 2.  (default: 2)
d (int): The depth multiplier (number of spatial filters)
num_electrodes (int): The number of electrodes
input_time_length (int): [sec]
sampling rate (int):
dropout (float): Probability of an element to be zeroed in the dropout layers. (default: 0.25)
"""
import torch
import torch.nn as nn
import numpy as np
from Model.Base.Layer import LinearWithConstraint, Conv2dWithConstraint, np_to_tensor, FocalLoss


class EEGNet(nn.Module):
    def __init__(
            self,
            f1: int,
            f2: int,
            d: int,
            num_electrodes: int,
            input_time_length: int,
            sampling_rate: int,
            dropout_rate: float,
            num_classes: int
    ):
        super(EEGNet, self).__init__()

        temporal_kernel_size = int(sampling_rate//2)

        # Conv2D
        self.layer1 = nn.Sequential(
            Conv2dWithConstraint(
                in_channels=1,
                out_channels=f1,
                kernel_size=(1, temporal_kernel_size),
                bias=False,
                padding=(0, int(temporal_kernel_size // 2))
            ),
            nn.BatchNorm2d(num_features=f1, momentum=0.01, affine=True, eps=1e-3),
        )

        # DepthWiseConv2D
        self.layer2 = nn.Sequential(
            Conv2dWithConstraint(
                in_channels=f1,
                out_channels=f1 * d,
                kernel_size=(num_electrodes, 1),
                bias=False,
                padding=(0, 0),
                groups=f1,
                max_norm=1
            ),
            nn.BatchNorm2d(num_features=f1 * d, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(dropout_rate)

        )

        # SeparableConv2D
        self.layer3 = nn.Sequential(
            # SeparableDepth
            Conv2dWithConstraint(
                in_channels=f1*d,
                out_channels=f1*d,
                kernel_size=(1, 16),
                bias=False,
                groups=f1*d,
                padding=(0, 16 // 2)
            ),
            # SeparablePoint
            Conv2dWithConstraint(
                in_channels=f1*d,
                out_channels=f2,
                kernel_size=(1, 1),
                bias=False,
                padding=(0, 0)
            ),
            nn.BatchNorm2d(num_features=f2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Dropout(dropout_rate)
        )

        out = self.layer1(
            np_to_tensor(
                np.ones(
                    (1, 1, num_electrodes, input_time_length*sampling_rate),
                    dtype=np.float32
                )
            )
        )
        out = self.layer2(out)
        out = self.layer3(out)
        final_length = out.reshape(-1).shape[0]

        self.fc = nn.Sequential(
            nn.Flatten(),
            LinearWithConstraint(in_features=final_length, out_features=num_classes, max_norm=0.25)
        )

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        out = self.fc(x)
        return out
