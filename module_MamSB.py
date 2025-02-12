import torch
from torch import nn
from component.conv_modules import *
import torch.nn.functional as F
import torch.fft
from component.Local_CNN_Branch import *
from module_mamba import NMOModel

class MamSB(nn.Module):
    def __init__(self, shape_in, num_interactions=3):
        super(MamSB, self).__init__()
        T, C, H, W = shape_in
        self.lc_block = Local_CNN_Branch(in_channels=C, out_channels=C)
        self.nmo_block = NMOModel(shape_in)

        self.up = nn.ConvTranspose2d(C, C, kernel_size=3, stride=1, padding=1)
        self.down = nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1)
        self.conv1x1 = nn.Conv2d(C, C, kernel_size=1)
        self.num_interactions = num_interactions

    def forward(self, x_raw):
        B, T, C, H, W = x_raw.shape
        nmo_output, nmo_features = self.nmo_block(x_raw)
        lc_features = self.lc_block(x_raw)

        for _ in range(self.num_interactions):
            nmo_features_up = self.up(nmo_features.view(B * T, C, H, W)).view(B, T, C, H, W)
            lc_features = self.conv1x1(lc_features.view(B * T, C, H, W)).view(B, T, C, H, W)
            combined_features = nmo_features_up + lc_features

            nmo_output, nmo_features = self.nmo_block(combined_features)
            lc_features = self.lc_block(combined_features)

            nmo_features_down = self.down(nmo_features.view(B * T, C, H, W)).view(B, T, C, H, W)
            lc_features = self.conv1x1(lc_features.view(B * T, C, H, W)).view(B, T, C, H, W)
            combined_features = nmo_features_down + lc_features

            nmo_output, nmo_features = self.nmo_block(combined_features)
            lc_features = self.lc_block(combined_features)

        return nmo_features + lc_features


class MamSB_woCNN(nn.Module):
    def __init__(self, shape_in, num_interactions=3):
        super(MamSB_woCNN, self).__init__()
        T, C, H, W = shape_in
        self.lc_block = Local_CNN_Branch(in_channels=C, out_channels=C)
        self.nmo_block = NMOModel(shape_in)

        self.up = nn.ConvTranspose2d(C, C, kernel_size=3, stride=1, padding=1)
        self.down = nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1)
        self.conv1x1 = nn.Conv2d(C, C, kernel_size=1)
        self.num_interactions = num_interactions

    def forward(self, x_raw):
        B, T, C, H, W = x_raw.shape
        nmo_output, nmo_features = self.nmo_block(x_raw)
        lc_features = self.lc_block(x_raw)

        for _ in range(self.num_interactions):
            nmo_features_up = self.up(nmo_features.view(B * T, C, H, W)).view(B, T, C, H, W)
            lc_features = self.conv1x1(lc_features.view(B * T, C, H, W)).view(B, T, C, H, W)
            combined_features = nmo_features_up + lc_features

            nmo_output, nmo_features = self.nmo_block(combined_features)
            lc_features = self.lc_block(combined_features)

            nmo_features_down = self.down(nmo_features.view(B * T, C, H, W)).view(B, T, C, H, W)
            lc_features = self.conv1x1(lc_features.view(B * T, C, H, W)).view(B, T, C, H, W)
            combined_features = nmo_features_down + lc_features

            nmo_output, nmo_features = self.nmo_block(combined_features)
            lc_features = self.lc_block(combined_features)

        return nmo_features

class MamSB_woMamba(nn.Module):
    def __init__(self, shape_in, num_interactions=3):
        super(MamSB_woMamba, self).__init__()
        T, C, H, W = shape_in
        self.lc_block = Local_CNN_Branch(in_channels=C, out_channels=C)
        self.nmo_block = NMOModel(shape_in)

        self.up = nn.ConvTranspose2d(C, C, kernel_size=3, stride=1, padding=1)
        self.down = nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1)
        self.conv1x1 = nn.Conv2d(C, C, kernel_size=1)
        self.num_interactions = num_interactions

    def forward(self, x_raw):
        B, T, C, H, W = x_raw.shape
        nmo_output, nmo_features = self.nmo_block(x_raw)
        lc_features = self.lc_block(x_raw)

        for _ in range(self.num_interactions):
            nmo_features_up = self.up(nmo_features.view(B * T, C, H, W)).view(B, T, C, H, W)
            lc_features = self.conv1x1(lc_features.view(B * T, C, H, W)).view(B, T, C, H, W)
            combined_features = nmo_features_up + lc_features

            nmo_output, nmo_features = self.nmo_block(combined_features)
            lc_features = self.lc_block(combined_features)

            nmo_features_down = self.down(nmo_features.view(B * T, C, H, W)).view(B, T, C, H, W)
            lc_features = self.conv1x1(lc_features.view(B * T, C, H, W)).view(B, T, C, H, W)
            combined_features = nmo_features_down + lc_features

            nmo_output, nmo_features = self.nmo_block(combined_features)
            lc_features = self.lc_block(combined_features)

        return lc_features



if __name__ == '__main__':
    shape_in = (10, 1, 64, 64)
    x_raw = torch.randn(1, *shape_in)
    model = MamSB_woCNN(shape_in)

    finally_output = model(x_raw)
    print(finally_output.shape)

