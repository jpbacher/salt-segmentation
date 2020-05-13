import torch
import torch.nn as nn
from torch.nn import Sequential, Conv2d, ConvTranspose2d, MaxPool2d, ReLU
from torch.nn import functional as F
from torchvision.models import vgg11


def conv3x3(in_channels, out_channels):
    return Conv2d(in_channels, out_channels, kernel_size=3, padding=1)


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
        self.bottleneck = Decoder(
            num_filters * 8 * 2, num_filters * 8 * 2, num_filters * 8)
        self.dec5 = Decoder(
            num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 8)
        self.dec4 = Decoder(
            num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 4
        )
        self.dec3 = Decoder(
            num_filters * (8 + 4), num_filters * 4 * 2, num_filters * 2
        )
        self.dec2 = Decoder(
            num_filters * (4 + 2), num_filters * 2 * 2, num_filters
        )
        self.dec1 = ConvRelu(num_filters * (2 + 1), num_filters)
        self.out = Conv2d(num_filters, 1, kernel_size=1)

    def forward(self, x):
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(self.pool(conv1)))
        conv3s = self.relu(self.conv3s(self.pool(conv2)))
        conv3 = self.relu(self.conv3(conv3s))
        conv4s = self.relu(self.conv4s(self.pool(conv3)))
        conv4 = self.relu(self.conv4(conv4s))
        conv5s = self.relu(self.conv5s(self.pool(conv4)))
        conv5 = self.relu(self.conv5(conv5s))
        bottleneck = self.bottleneck(self.pool(conv5))
        dec5 = self.dec5(torch.cat([bottleneck, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        return F.sigmoid(self.out(dec1))
