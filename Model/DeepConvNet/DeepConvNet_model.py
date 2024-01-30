# -*- coding:utf-8 -*-
"""
Created on Mon Jan. 15 16:56:55 2024
@author: PJS

** Parameters **
num_electrodes (int): The number of electrodes
input_time_length (int): [sec]
sampling rate (int):
dropout (float): Probability of an element to be zeroed in the dropout layers. (default: 0.25)
kernel_length (int):
pool_size (int):
block_out_channels (List[int]):
"""
import torch
import torch.nn as nn
import numpy as np
from Model.Base.Layer import Conv2dWithConstraint, np_to_tensor, default_layer
from Config.model_config import DeepConvNetConfig


class DeepConvNet(nn.Module):
    def __init__(
            self,
            model_config: DeepConvNetConfig
    ):
        super(DeepConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            Conv2dWithConstraint(
                in_channels=1,
                out_channels=model_config.block_out_channels[0],
                kernel_size=(1, model_config.kernel_length),
                stride=1,
                bias=False,
                max_norm=2,
            ),
            Conv2dWithConstraint(
                in_channels=model_config.block_out_channels[0],
                out_channels=model_config.block_out_channels[0],
                kernel_size=(model_config.num_electrodes, 1),
                stride=1,
                bias=False,
                max_norm=2
            ),
            nn.BatchNorm2d(model_config.block_out_channels[0], momentum=0.1, eps=1e-05),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, model_config.pool_size), stride=(1, model_config.pool_size)),
            nn.Dropout(model_config.dropout_rate)
        )

        self.deep_layer = nn.ModuleList(
            [default_layer(in_channels=model_config.block_out_channels[i], out_channels=model_config.block_out_channels[i + 1],
                           kernel_length=model_config.kernel_length, pool_size=model_config.pool_size) for i in range(3)]
        )

        out = self.layer1(
            np_to_tensor(
                np.ones(
                    (1, 1, model_config.num_electrodes, model_config.input_time_length * model_config.sampling_rate),
                    dtype=np.float32
                )
            )
        )
        for layer in self.deep_layer:
            out = layer(out)
        final_length = out.reshape(-1).shape[0]

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=final_length, out_features=model_config.num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = self.layer1(x)
        for layer in self.deep_layer:
            x = layer(x)
        out = self.fc(x)
        return out
