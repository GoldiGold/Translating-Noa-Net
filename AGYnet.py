import torch
import os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import datasets, transforms
import numpy as np
from omri_new.config import CONF

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


def calc_padding(mri_dims, kernel_size=2):
    # the aim of this function is that the output size to be the same as input for stride convolution
    # this function calculate a padding for stride = 2 Conv3D
    # assuming stride = 2, kernel size = 2, and even tensor size
    # checked correctness for size:
    # Sheba75
    # MRI_DIMS = (128, 140, 112)
    # MRI_DIMS = (128, 160, 128)

    # HCP105
    # MRI_DIMS = (112, 140, 112)
    # MRI_DIMS = (128, 144, 128)

    # different size of stride, kernel or tensor size are not guaranteed to work
    # for more info about input/output size: https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html

    D, H, W = mri_dims
    D_pad = (int)(((D + kernel_size) - 2) * 0.5)
    H_pad = (int)(((H + kernel_size) - 2) * 0.5)
    W_pad = (int)(((W + kernel_size) - 2) * 0.5)
    out = (D_pad, H_pad, W_pad)
    return out


# model = Encoder().to(device)
# print(model)
# n = sum(param.numel() for param in model.parameters() if param.requires_grad)
# print(f"Number of parameters: {n}")


class Encoder(nn.Module):
    def __init__(self, mri_dims):
        super(Encoder, self).__init__()
        k = 3

        # layer1
        self.layer1a = nn.Sequential(torch.nn.Conv3d(1, CONF['ynet_ch'][0], k, stride=1, padding='same'), nn.PReLU())
        self.layer1b = nn.Sequential(
            torch.nn.Conv3d(CONF['ynet_ch'][0], CONF['ynet_ch'][1], 2, stride=2, padding=calc_padding(mri_dims)),
            nn.PReLU(), torch.nn.Dropout(p=0.5, inplace=False))

        # layer2
        self.layer2a = nn.Sequential(
            torch.nn.Conv3d(CONF['ynet_ch'][1], CONF['ynet_ch'][1], k, stride=1, padding='same'), nn.PReLU())
        self.layer2b = torch.nn.Conv3d(CONF['ynet_ch'][1], CONF['ynet_ch'][1], k, stride=1, padding='same')
        # sum
        self.layer2c = nn.PReLU()
        self.layer2d = nn.Sequential(
            torch.nn.Conv3d(CONF['ynet_ch'][1], CONF['ynet_ch'][2], 2, stride=2, padding=calc_padding(mri_dims)),
            nn.PReLU(), torch.nn.Dropout(p=0.5, inplace=False))

        # layer3
        self.layer3a = nn.Sequential(
            torch.nn.Conv3d(CONF['ynet_ch'][2], CONF['ynet_ch'][2], k, stride=1, padding='same'), nn.PReLU())
        self.layer3b = nn.Sequential(
            torch.nn.Conv3d(CONF['ynet_ch'][2], CONF['ynet_ch'][2], k, stride=1, padding='same'), nn.PReLU()
            , torch.nn.Conv3d(CONF['ynet_ch'][2], CONF['ynet_ch'][2], k, stride=1, padding='same'))
        # sum
        self.layer3c = nn.PReLU()
        self.layer3d = nn.Sequential(
            torch.nn.Conv3d(CONF['ynet_ch'][2], CONF['ynet_ch'][3], 2, stride=2, padding=calc_padding(mri_dims)),
            nn.PReLU(), torch.nn.Dropout(p=0.5, inplace=False))

        # layer4
        self.layer4a = nn.Sequential(
            torch.nn.Conv3d(CONF['ynet_ch'][3], CONF['ynet_ch'][3], k, stride=1, padding='same'), nn.PReLU())
        self.layer4b = nn.Sequential(
            torch.nn.Conv3d(CONF['ynet_ch'][3], CONF['ynet_ch'][3], k, stride=1, padding='same'), nn.PReLU()
            , torch.nn.Conv3d(CONF['ynet_ch'][3], CONF['ynet_ch'][3], k, stride=1, padding='same'))
        # sum
        self.layer4c = nn.PReLU()
        self.layer4d = nn.Sequential(
            torch.nn.Conv3d(CONF['ynet_ch'][3], CONF['ynet_ch'][4], 2, stride=2, padding=calc_padding(mri_dims)),
            nn.PReLU(), torch.nn.Dropout(p=0.5, inplace=False))

        # layers5
        self.layer5a = nn.Sequential(
            torch.nn.Conv3d(CONF['ynet_ch'][4], CONF['ynet_ch'][4], k, stride=1, padding='same'), nn.PReLU())
        self.layer5b = nn.Sequential(
            torch.nn.Conv3d(CONF['ynet_ch'][4], CONF['ynet_ch'][4], k, stride=1, padding='same'), nn.PReLU()
            , torch.nn.Conv3d(CONF['ynet_ch'][4], CONF['ynet_ch'][4], k, stride=1, padding='same'))
        # sum
        self.layer5c = nn.PReLU()
        self.layer5d = torch.nn.ConvTranspose3d(CONF['ynet_ch'][4], CONF['ynet_ch'][2], 2, stride=2,
                                                padding=calc_padding(mri_dims))
        self.layer5E = nn.Sequential(nn.PReLU(), torch.nn.Dropout(p=0.5, inplace=False))

    def forward(self, input):
        L1_f = self.layer1a(input)
        L1b = self.layer1b(L1_f)

        L2a = self.layer2a(L1b)
        L2b = self.layer2b(L2a)
        L2_sum = L2b + L1b
        L2_f = self.layer2c(L2_sum)
        L2d = self.layer2d(L2_f)

        L3a = self.layer3a(L2d)
        L3b = self.layer3b(L3a)
        L3_sum = L3b + L2d
        L3_f = self.layer3c(L3_sum)
        L3d = self.layer3d(L3_f)

        L4a = self.layer4a(L3d)
        L4b = self.layer4b(L4a)
        L4_sum = L4b + L3d
        L4_f = self.layer4c(L4_sum)
        L4d = self.layer4d(L4_f)

        L5a = self.layer5a(L4d)
        L5b = self.layer5b(L5a)
        L5_sum = L4d + L5b
        L5_f_prelu5 = self.layer5c(L5_sum)
        L5_f = self.layer5d(L5_f_prelu5)
        L5_f_MID = self.layer5E(L5_f)

        return L1_f, L2_f, L3_f, L4_f, L5_f, L5_f_prelu5, L5_f_MID


# T1 = torch.tensor(np.random.rand(1,40,44,48).astype(float),dtype = torch.float).to(device)
# model = Encoder(mri_dims=(40,44,48)).to(device)
# 'ynet_ch'            : [20, 40, 80, 160, 320]
# L1_f,L2_f,L3_f,L4_f,L5_f,L5_f_prelu5,L5_f_MID = model(T1)
# print(L5_f.shape)


# model = Decoder().to(device)
# print(model)
# n = sum(param.numel() for param in model.parameters() if param.requires_grad)
# print(f"Number of parameters: {n}")


class AtentionGate3Inputs(nn.Module):
    def __init__(self, in_channels, in_shape, return_mask=False, mri_dims=(128, 144, 128)):
        super(AtentionGate3Inputs, self).__init__()
        self.in_shape = in_shape
        self.return_mask = return_mask
        self.mri_dims = mri_dims
        inter_channels = in_channels // 2
        if inter_channels == 0:
            inter_channels = 1
        # self.Wx = torch.nn.Conv3d(inter_channels,inter_channels,2,stride=2,padding='valid')
        # self.Wg = torch.nn.Conv3d(inter_channels,inter_channels,1,stride=1,padding='valid',bias=True)
        self.Wx = torch.nn.Conv3d(inter_channels // 2, inter_channels, 2, stride=2, padding=calc_padding(mri_dims))
        self.Wg = torch.nn.Conv3d(inter_channels * 2, inter_channels, 1, stride=1, padding='same', bias=True)
        # self.path = nn.Sequential(nn.ReLU(),torch.nn.Conv3d(1,1,kernel_size=1,stride=1,padding='valid',bias=True),nn.Sigmoid())
        self.path = nn.Sequential(nn.ReLU(), torch.nn.Conv3d(inter_channels, 1, kernel_size=1, stride=1, padding='same',
                                                             bias=True), nn.Sigmoid())
        # UPSAMPLING
        # ELEMENTWIZE MULTIPLICATION
        # self.Wy = torch.nn.Conv3d(in_channels,in_channels,kernel_size=1,stride=1,padding='valid',bias=True)
        self.Wy = torch.nn.Conv3d(inter_channels // 2, inter_channels // 2, kernel_size=1, stride=1, padding='same',
                                  bias=True)
        # ELEMENTWIZE SUM

    def forward(self, X1, X2, G):
        WX = self.Wx(X2)
        WG = self.Wg(G)
        print(f"WX: {WX.shape}, WG {WG.shape}")
        SUM = WX + WG
        PATH = self.path(SUM)
        print(f"PATH: {PATH.shape}")
        # UPSAMPALED = torch.nn.functional.upsample(PATH, size=[CONF['batch_size'], 1, *self.in_shape], scale_factor=2, mode='trilinear') #SHOLUD NOT WORK TRILINEAR
        # UPSAMPALED = torch.nn.functional.upsample(PATH, size=[ 1, *self.mri_dims], mode='bicubic')  # SHOLUD NOT WORK TRILINEAR
        # print(f"UPSAMPALED {UPSAMPALED.shape}, X1 {X1.shape}")
        # PRODUCT = UPSAMPALED*X1
        UPSAMPALED = None
        PRODUCT = PATH * X1
        CONV = self.Wy(PRODUCT)
        RES = CONV + X1
        if self.return_mask:
            output = UPSAMPALED
        else:
            output = RES
        return output


# mri_dims=(128,144,28)
# model = AtentionGate3Inputs(in_channels=CONF['ynet_ch'][2],in_shape=(int(mri_dims[0]/4),int(mri_dims[1]/4),int(mri_dims[2]/4)),return_mask=False).to(device)
# print(model)
# n = sum(param.numel() for param in model.parameters() if param.requires_grad)
# print(f"Number of parameters: {n}")

# 'ynet_ch'            : [20, 40, 80, 160, 320]
class Decoder(nn.Module):
    def __init__(self, mri_dims=(128, 144, 128)):
        super(Decoder, self).__init__()

        # TODO check correctness of input parameter CONF['ynet_ch'][i]
        # TODO is ch_in == ch_o ???
        self.mri_dims = mri_dims
        k = 3
        self.k = k

        # LAYER 4
        self.AG4_1 = AtentionGate3Inputs(in_channels=CONF['ynet_ch'][4], in_shape=(
        int(self.mri_dims[0] / 8), int(self.mri_dims[1] / 8), int(self.mri_dims[2] / 8)), mri_dims=mri_dims)
        self.AG4_2 = AtentionGate3Inputs(in_channels=CONF['ynet_ch'][4], in_shape=(
        int(self.mri_dims[0] / 8), int(self.mri_dims[1] / 8), int(self.mri_dims[2] / 8)), mri_dims=mri_dims)

        # self.layer4a = nn.Sequential(
        # torch.nn.Conv3d(CONF['ynet_ch'][3],CONF['ynet_ch'][3],k,stride=1,padding='same'),nn.PReLU(),
        # torch.nn.Conv3d(CONF['ynet_ch'][3], CONF['ynet_ch'][3], k, stride=1, padding='same'), nn.PReLU(),
        # torch.nn.Conv3d(CONF['ynet_ch'][3], CONF['ynet_ch'][3], k, stride=1, padding='same'))
        self.layer4a = nn.Sequential(
            torch.nn.Conv3d(CONF['ynet_ch'][4], CONF['ynet_ch'][3], k, stride=1, padding='same'), nn.PReLU(),
            torch.nn.Conv3d(CONF['ynet_ch'][3], CONF['ynet_ch'][3], k, stride=1, padding='same'), nn.PReLU(),
            torch.nn.Conv3d(CONF['ynet_ch'][3], CONF['ynet_ch'][3], k, stride=1, padding='same'))

        # ADD
        self.layer4b = nn.PReLU()
        self.layer4c = nn.Sequential(
            torch.nn.ConvTranspose3d(CONF['ynet_ch'][3], CONF['ynet_ch'][3], 2, stride=2,
                                     padding=calc_padding(mri_dims)),
            nn.PReLU(), torch.nn.Dropout(p=0.5, inplace=False))

        # LAYER 3

        self.AG3_1 = AtentionGate3Inputs(in_channels=CONF['ynet_ch'][4], in_shape=(
        int(self.mri_dims[0] / 4), int(self.mri_dims[1] / 4), int(self.mri_dims[2] / 4)), mri_dims=mri_dims)
        self.AG3_2 = AtentionGate3Inputs(in_channels=CONF['ynet_ch'][4], in_shape=(
        int(self.mri_dims[0] / 4), int(self.mri_dims[1] / 4), int(self.mri_dims[2] / 4)), mri_dims=mri_dims)

        self.layer3a = nn.Sequential(
            torch.nn.Conv3d(CONF['ynet_ch'][4], CONF['ynet_ch'][3], k, stride=1, padding='same'), nn.PReLU(),
            torch.nn.Conv3d(CONF['ynet_ch'][3], CONF['ynet_ch'][3], k, stride=1, padding='same'), nn.PReLU(),
            torch.nn.Conv3d(CONF['ynet_ch'][3], CONF['ynet_ch'][3], k, stride=1, padding='same'))
        self.layer3b = nn.PReLU()
        self.layer3c = nn.Sequential(
            torch.nn.ConvTranspose3d(CONF['ynet_ch'][3], CONF['ynet_ch'][2], 2, stride=2,
                                     padding=calc_padding(mri_dims)),
            nn.PReLU(), torch.nn.Dropout(p=0.5, inplace=False))

        # LAYER2

        self.AG2_1 = AtentionGate3Inputs(in_channels=CONF['ynet_ch'][2], in_shape=(
        int(self.mri_dims[0] / 2), int(self.mri_dims[1] / 2), int(self.mri_dims[2] / 2)), mri_dims=mri_dims)
        self.AG2_2 = AtentionGate3Inputs(in_channels=CONF['ynet_ch'][2], in_shape=(
        int(self.mri_dims[0] / 2), int(self.mri_dims[1] / 2), int(self.mri_dims[2] / 2)), mri_dims=mri_dims)

        self.layer2a = nn.Sequential(
            torch.nn.Conv3d(CONF['ynet_ch'][3], CONF['ynet_ch'][3], k, stride=1, padding='same'), nn.PReLU(),
            torch.nn.Conv3d(CONF['ynet_ch'][3], CONF['ynet_ch'][3], k, stride=1, padding='same'))
        self.layer2b = nn.PReLU()
        self.layer2c = nn.Sequential(
            torch.nn.ConvTranspose3d(CONF['ynet_ch'][2], CONF['ynet_ch'][1], 2, stride=2,
                                     padding=calc_padding(mri_dims)),
            nn.PReLU(), torch.nn.Dropout(p=0.5, inplace=False))

        # LAYER1

        self.AG1_1 = AtentionGate3Inputs(in_channels=CONF['ynet_ch'][0],
                                         in_shape=(self.mri_dims[0], self.mri_dims[1], self.mri_dims[2]),
                                         mri_dims=mri_dims)
        self.AG1_2 = AtentionGate3Inputs(in_channels=CONF['ynet_ch'][0],
                                         in_shape=(self.mri_dims[0], self.mri_dims[1], self.mri_dims[2]),
                                         mri_dims=mri_dims)

        self.layer1a = torch.nn.Conv3d(CONF['ynet_ch'][1], CONF['ynet_ch'][1], k, stride=1, padding='same')

        self.layer1b = nn.PReLU()
        self.layer1c = torch.nn.Conv3d(CONF['ynet_ch'][2], CONF['class_num'] + 1, 1, stride=1, padding='same')
        self.softmax = torch.nn.Softmax(dim=4)

    # def forward(self, L1_f_T1, L2_f_T1, L3_f_T1, L4_f_T1, L5_f_T1, L5_f_prelu5_T1, L5_f_MID_T1, L1_f_RGB, L2_f_RGB,L3_f_RGB, L4_f_RGB, L5_f_RGB, L5_f_prelu5_RGB, L5_f_MID_RGB):
    # def forward(self,L1_f_T1,L2_f_T1,L3_f_T1,L4_f_T1,L5_f_T1,L5_f_prelu5_T1,L5_f_MID_T1,L1_f_RGB,L2_f_RGB,L3_f_RGB,L4_f_RGB,L5_f_RGB,L5_f_prelu5_RGB,L5_f_MID_RGB):
    def forward(self, L1_f_T1, L2_f_T1, L3_f_T1, L5_f_T1, L5_f_prelu5_T1, L5_f_MID_T1, L1_f_RGB, L2_f_RGB, L3_f_RGB,
                L5_f_RGB, L5_f_prelu5_RGB, L5_f_MID_RGB):
        merge_mid_T1_RGB = torch.cat((L5_f_MID_T1, L5_f_MID_RGB), dim=0)
        print(f"L5_f_T1: {L5_f_T1.shape}, L5_f_RGB: {L5_f_RGB.shape}, L5_f_RGB: {L5_f_RGB.shape}")
        prelu_4_T1_fused = self.AG4_1(L5_f_T1, L5_f_RGB, L5_f_prelu5_RGB)
        prelu_4_RGB_fused = self.AG4_2(L5_f_RGB, L5_f_T1, L5_f_prelu5_T1)
        # prelu_4_T1_fused = self.AG4_1(L5_f_T1,L5_f_RGB,L5_f_RGB)
        # prelu_4_RGB_fused = self.AG4_2(L5_f_RGB,L5_f_T1,L5_f_T1)
        print("L4 is done")
        merge_mid_T1_RGB_with_fusion = torch.cat((prelu_4_T1_fused, prelu_4_RGB_fused, L5_f_MID_T1, L5_f_MID_RGB),
                                                 dim=0)
        wc_right_43_RGBT1 = self.layer4a(merge_mid_T1_RGB_with_fusion)
        add_4_RGBT1 = merge_mid_T1_RGB + wc_right_43_RGBT1
        dropout_right_4_RGBT1 = self.layer4c(self.layer4b(add_4_RGBT1))

        print(f"L3_f_T1: {L3_f_T1.shape}, L3_f_RGB: {L3_f_RGB.shape}, prelu_4_RGB_fused: {prelu_4_RGB_fused.shape}")
        prelu_3_T1_fused = self.AG3_1(L3_f_T1, L3_f_RGB, prelu_4_RGB_fused)
        prelu_3_RGB_fused = self.AG3_2(L3_f_RGB, L3_f_T1, prelu_4_T1_fused)
        merge_right_3_T1_RGB = torch.cat((dropout_right_4_RGBT1, prelu_3_T1_fused, prelu_3_RGB_fused), dim=0)
        wc_right_33_RGBT1 = self.layer3a(merge_right_3_T1_RGB)
        add_3_RGBT1 = wc_right_33_RGBT1 + dropout_right_4_RGBT1
        dropout_right_3_RGBT1 = self.layer3c(self.layer3b(add_3_RGBT1))
        print("L3 is done")

        print(f"L2_f_T1: {L2_f_T1.shape}, L2_f_RGB: {L2_f_RGB.shape}, L2_f_RGB: {prelu_3_RGB_fused.shape}")
        prelu_2_T1_fused = self.AG2_1(L2_f_T1, L2_f_RGB, prelu_3_RGB_fused)
        prelu_2_RGB_fused = self.AG2_2(L2_f_RGB, L2_f_T1, prelu_3_T1_fused)
        merge_right_2_T1_RGB = torch.cat((dropout_right_3_RGBT1, prelu_2_T1_fused, prelu_2_RGB_fused), dim=0)
        wc_right_22_RGBT1 = self.layer2a(merge_right_2_T1_RGB)
        add_2_RGBT1 = wc_right_22_RGBT1 + dropout_right_3_RGBT1
        dropout_right_2_RGBT1 = self.layer2c(self.layer2b(add_2_RGBT1))

        prelu_1_T1_fused = self.AG1_1(L1_f_T1, L1_f_RGB, prelu_2_RGB_fused)
        prelu_1_RGB_fused = self.AG1_1(L1_f_RGB, L1_f_T1, prelu_2_T1_fused)
        merge_right_1_T1_RGB = torch.cat((dropout_right_2_RGBT1, prelu_1_T1_fused, prelu_1_RGB_fused), dim=0)
        wc_right_11_RGBT1 = self.layer1a(merge_right_1_T1_RGB)
        add_1_RGBT1 = wc_right_11_RGBT1 + dropout_right_2_RGBT1
        wc_out2_RGBT1 = self.layer1c(self.layer1b(add_1_RGBT1))
        out = self.softmax(wc_out2_RGBT1)
        return out


# model = Encoder().to(device)
# print(model)
# n = sum(param.numel() for param in model.parameters() if param.requires_grad)
# print(f"Number of parameters: {n}")

class AGYnet_v2(nn.Module):
    def __init__(self, mri_dims=(128, 144, 128)):
        super(AGYnet_v2, self).__init__()
        self.T1_path = Encoder(mri_dims)
        self.RGB_path = Encoder(mri_dims)
        self.RGBT1_path = Decoder(mri_dims)

    def forward(self, T1, RGB):
        L1_f_T1, L2_f_T1, L3_f_T1, L4_f_T1, L5_f_T1, L5_f_prelu5_T1, L5_f_MID_T1 = self.T1_path(T1)
        L1_f_RGB, L2_f_RGB, L3_f_RGB, L4_f_RGB, L5_f_RGB, L5_f_prelu5_RGB, L5_f_MID_RGB = self.RGB_path(RGB)
        out = self.RGBT1_path(L1_f_T1, L2_f_T1, L3_f_T1, L5_f_T1, L5_f_prelu5_T1, L5_f_MID_T1, L1_f_RGB, L2_f_RGB,
                              L3_f_RGB, L5_f_RGB, L5_f_prelu5_RGB, L5_f_MID_RGB)
        return out


# model = AGYnet_v2().to(device)
# print(model)
# n = sum(param.numel() for param in model.parameters() if param.requires_grad)
# print(f"Number of parameters: {n}")


class test(nn.Module):
    def __init__(self):
        super(test, self).__init__()
        mri_dims = (40, 44, 48)
        self.layer = torch.nn.Conv3d(1, 1, 2, 2, padding=calc_padding(mri_dims, 2))

    def forward(self, img):
        return self.layer(img)

if __name__ == 'main':
    T1 = torch.tensor(np.random.rand(1, 40, 48, 56).astype(float), dtype=torch.float).to(device)
    RGB = torch.tensor(np.random.rand(1, 40, 48, 56).astype(float), dtype=torch.float).to(device)
    # model = test().to(device)
    # n = sum(param.numel() for param in model.parameters() if param.requires_grad)
    # print(f"Number of parameters: {n}")
    # out = model(T1)
    # print(out.shape)
    model = AGYnet_v2(mri_dims=(40, 48, 56)).to(device)
    # n = sum(param.numel() for param in model.parameters() if param.requires_grad)
    # print(f"Number of parameters: {n}")
    out = model(T1, RGB)
    print(out.shape)

    # T1 = torch.tensor(np.random.rand(1,40,44,48).astype(float),dtype = torch.float).to(device)
    # model = Decoder(mri_dims=(40,44,48)).to(device)
    # 'ynet_ch'            : [20, 40, 80, 160, 320]
    # out = model(T1)
    # print(out.shape)
