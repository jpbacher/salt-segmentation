import torch
import torch.nn as nn
from torch.nn import Sequential, Conv2d, ConvTranspose2d, MaxPool2d, ReLU
from torch.nn import functional as F
from torchvision.models import vgg11


def conv3x3(in_, out):
    return Conv2d(in_, out, kernel_size=3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = conv3x3(in_channels, out_channels)
        self.activation = ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        return self.activation(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.block = Sequential(
            ConvRelu(in_channels, mid_channels),
            ConvTranspose2d(mid_channels, out_channels, kernel_size=3, stride=2,
                            padding=1, output_padding=1),
            ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UnetVgg11(nn.Module):
    def __init(self, num_filters=32):
        super().__init__()
        self.pool = MaxPool2d(kernel_size=2, stride=2)
        self.encoder = vgg11().features
        self.relu = self.encoder[1]
        self.conv1 = self.encoder[0]
        self.conv2 = self.encoder[3]
        self.conv3s = self.encoder[6]
        self.conv3 = self.encoder[8]
        self.conv4s = self.encoder[11]
        self.conv4 = self.encoder[13]
        self.conv5s = self.encoder[16]
        self.conv5 = self.encoder[18]

