from __future__ import annotations

import torch
from torch import nn


class LeNetBackbone(nn.Module):
    """
    LeNet-style backbone for 256x256 grayscale MRI slices.

    Same building blocks as LeNet-5 (convolution, ReLU, max pooling),
    stacked deeper so each output cell sees enough context to tell
    the five structures apart.

    Input:
        [batch, 1, 256, 256]

    Output:
        [batch, 96, 32, 32]   (stride 8)
    """

    out_channels = 96

    def __init__(self) -> None:
        super().__init__()

        # 256x256x1 -> 256x256x8
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=True,
        )

        # 256x256x8 -> 128x128x8
        self.pool1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        # 128x128x8 -> 128x128x24
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=24,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=True,
        )

        # 128x128x24 -> 64x64x24
        self.pool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        # 64x64x24 -> 64x64x48
        self.conv3 = nn.Conv2d(
            in_channels=24,
            out_channels=48,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        # 64x64x48 -> 32x32x48
        self.pool3 = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        # 32x32x48 -> 32x32x96
        self.conv4 = nn.Conv2d(
            in_channels=48,
            out_channels=96,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        # 32x32x96 -> 32x32x96
        self.conv5 = nn.Conv2d(
            in_channels=96,
            out_channels=96,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        self.relu = nn.ReLU(inplace=False)

    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv1(images)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.relu(x)

        return x