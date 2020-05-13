import torch
import torch.nn as nn
from torch.nn import Conv2d, ReLU
from torch.nn import functional as F
from torchvision.models import vgg11


def conv3x3(in_, out):
    return Conv2d(in_, out, kernel_size=3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        return self.activation(x)
