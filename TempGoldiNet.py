import torch
import torchvision
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch import nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import nibabel as nib
import os
import consts


class TempGoldiNet(nn.Module):
    def __init__(self, conv3d_channels: list, conv3d_strides: list):
        super(TempGoldiNet, self).__init__()
        self.conv3d_layers = []
        # First channel should b 1 because we have a single 3D channel (a single grayscale image)
        for idx in range(len(conv3d_channels) - 1):
            self.conv3d_layers.append(
                nn.Conv3d(conv3d_channels[idx], conv3d_channels[idx + 1], kernel_size=(2, 2, 2),
                          stride=conv3d_strides[idx]))
        self.fully_connected = nn.Linear(1000, 7)

    def forward(self, brains):
        for layer in self.conv3d_layers:
            brains = layer(brains)
        return brains
