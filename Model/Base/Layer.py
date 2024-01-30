# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LinearWithConstraint(nn.Linear):
    def __init__(self, *config, max_norm=1, **kwconfig):
        self.max_norm = max_norm
        super(LinearWithConstraint, self).__init__(*config, **kwconfig)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(LinearWithConstraint, self).forward(x)


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *config, max_norm=1, **kwconfig):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*config, **kwconfig)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)


def np_to_tensor(x, requires_grad=False, dtype=None, pin_memory=False, **tensor_kwargs):
    if not hasattr(x, "__len__"):
        x = [x]
    x = np.asarray(x)
    if dtype is not None:
        x = x.astype(dtype)
    x_tensor = torch.tensor(x, requires_grad=requires_grad, **tensor_kwargs)
    if pin_memory:
        x_tensor = x_tensor.pin_memory()
    return x_tensor


# DeepConvNet default block
def default_layer(in_channels, out_channels, kernel_length, pool_size):
    default_layer_ = nn.Sequential(
        Conv2dWithConstraint(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, kernel_length),
            stride=1,
            bias=False,
            max_norm=2
        ),
        nn.BatchNorm2d(out_channels, momentum=0.1, eps=1e-05),
        nn.ELU(),
        nn.MaxPool2d(kernel_size=(1, pool_size), stride=(1, pool_size)),
        nn.Dropout(0.5)
    )
    return default_layer_


# ShallowConvNet
class ActSquare(nn.Module):
    def __init__(self):
        super(ActSquare, self).__init__()
        pass

    def forward(self, x):
        return torch.square(x)


# ShallowConvNet
class ActLog(nn.Module):
    def __init__(self, eps=1e-06):
        super(ActLog, self).__init__()
        self.eps = eps

    def forward(self, x):
        return torch.log(torch.clamp(x, min=self.eps))


# Loss function
class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2, reduction: str = 'mean'):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction=self.reduction, weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss