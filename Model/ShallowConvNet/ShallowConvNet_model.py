# -*- coding:utf-8 -*-
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
from Model.Base.Layer import Conv2dWithConstraint, ActSquare, ActLog, np_to_tensor


class ShallowConvNet(nn.Module):
    def __init__(
            self,
            num_electrodes: int,
            input_time_length: int,
            sampling_rate: int,
            dropout_rate: float,
            num_classes: int,
            kernel_length_1: int = None,
            f1: int = None,
            f2: int = None
    ):

        super(ShallowConvNet, self).__init__()
        if f1 is None:
            f1 = 40
        if f2 is None:
            f2 = 40
        if kernel_length_1 is None:
            kernel_length_1 = 25
        kernel_length_2 = kernel_length_1 * 3

        self.layer1 = nn.Sequential(
            Conv2dWithConstraint(
                in_channels=1,
                out_channels=f1,
                kernel_size=(1, kernel_length_1),
                stride=1
            ),
            Conv2dWithConstraint(
                in_channels=f1,
                out_channels=f2,
                kernel_size=(num_electrodes, 1),
                bias=False,
                stride=1,
                # groups=f1
            ),
            nn.BatchNorm2d(f2, momentum=0.1, affine=True),
            ActSquare(),
            nn.AvgPool2d(kernel_size=(1, kernel_length_2), stride=(1, kernel_length_2//5)),
            ActLog(),
            nn.Dropout(dropout_rate)
        )

        out = self.layer1(
            np_to_tensor(
                np.ones(
                    (1, 1, num_electrodes, input_time_length * sampling_rate),
                    dtype=np.float32
                )
            )
        )
        final_length = out.reshape(-1).shape[0]

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=final_length, out_features=num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = self.layer1(x)
        out = self.fc(x)
        return out
