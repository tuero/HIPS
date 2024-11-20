import torch
import torch.nn as nn
import torch.nn.functional as F


# Simple MLP network
class MLP(nn.Module):
    def __init__(self, input_size: int, layer_sizes: list[int], output_size: int):
        super(MLP, self).__init__()
        _layer_sizes = [input_size] + layer_sizes
        layers: list[nn.Module] = []
        for i in range(1, len(_layer_sizes)):
            layers.append(nn.Linear(_layer_sizes[i - 1], _layer_sizes[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(_layer_sizes[-1], output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def conv1x1(in_channels: int, out_channels: int) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels, out_channels, 1, stride=1, padding=0, bias=True, dilation=1
    )


# Residual Block
class ResidualHead(nn.Module):
    def __init__(
        self, input_channels: int, output_channels: int, use_batchnorm: bool = False
    ):
        super(ResidualHead, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, 3, stride=1, padding=1)
        self.use_batchnorm = use_batchnorm
        if self.use_batchnorm:
            self.bn = nn.BatchNorm2d(output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.use_batchnorm:
            out = self.bn(out)
        out = F.relu(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, num_channels: int, use_batchnorm: bool = False):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, stride=1, padding=1)
        self.use_batchnorm = use_batchnorm
        if self.use_batchnorm:
            self.bn1 = nn.BatchNorm2d(num_channels)
            self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        if self.use_batchnorm:
            out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        if self.use_batchnorm:
            out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out
