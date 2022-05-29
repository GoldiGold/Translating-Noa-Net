import torch
import os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import datasets, transforms
import numpy as np
from config import CONF

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


def calc_padding(mri_dims, kernel_size=2):
    D, H, W = mri_dims
    D_pad = (int)(((D + kernel_size) - 2) * 0.5)
    H_pad = (int)(((H + kernel_size) - 2) * 0.5)
    W_pad = (int)(((W + kernel_size) - 2) * 0.5)
    out = (D_pad, H_pad, W_pad)
    return out


class test(nn.Module):
    def __init__(self):
        super(test, self).__init__()
        mri_dims = (40, 44, 48)
        print(calc_padding(mri_dims, 2))
        self.layer = torch.nn.Conv3d(1, 3, 2, 2, padding=calc_padding(mri_dims, 2))

    def forward(self, img):
        return self.layer(img)


T1 = torch.tensor(np.random.rand(1, 40, 44, 48).astype(float), dtype=torch.float).to(device)
# RGB = torch.tensor(np.random.rand(1,1,48,48,48).astype(float)).to(device)
model = test().to(device)
# n = sum(param.numel() for param in model.parameters() if param.requires_grad)
# print(f"Number of parameters: {n}")

# 'ynet_ch'            : [20, 40, 80, 160, 320]
out = model(T1)
print(out.shape)
