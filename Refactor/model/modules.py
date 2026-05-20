"""Reusable convolution and multi-scale blocks for SG_LLIE."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Single convolution layer used when no activation should be applied."""

    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        """Create a 2D convolution from `in_channel` to `out_channel`."""
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
            dilation=dilation_rate,
        )

    def forward(self, x):
        """Apply convolution to BCHW tensor `x` and return BCHW features."""
        return self.conv(x)


class ConvReLUBlock(nn.Module):
    """Convolution followed by an in-place ReLU activation."""

    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        """Create a conv-ReLU block from `in_channel` to `out_channel`."""
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=True,
                dilation=dilation_rate,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Apply convolution and ReLU to BCHW tensor `x` and return BCHW features."""
        return self.conv(x)


class DenseBlock(nn.Module):
    """Dense dilated convolution block without outer residual addition."""

    def __init__(self, in_channel, d_list, inter_num):
        """Create dense conv layers using dilation rates from `d_list`."""
        super().__init__()
        self.conv_layers = nn.ModuleList()
        channels = in_channel
        for dilation_rate in d_list:
            self.conv_layers.append(
                ConvReLUBlock(channels, inter_num, 3, dilation_rate=dilation_rate, padding=dilation_rate)
            )
            channels += inter_num
        self.conv_post = ConvBlock(channels, in_channel, 1)

    def forward(self, x):
        """Process BCHW tensor `x` through dense layers and return BCHW features."""
        features = x
        for conv_layer in self.conv_layers:
            new_features = conv_layer(features)
            features = torch.cat([new_features, features], dim=1)
        return self.conv_post(features)


class ResidualDenseBlock(nn.Module):
    """Dense dilated convolution block with residual output."""

    def __init__(self, in_channel, d_list, inter_num):
        """Create dense conv layers and a residual projection back to `in_channel`."""
        super().__init__()
        self.body = DenseBlock(in_channel, d_list, inter_num)

    def forward(self, x):
        """Return `x` plus dense-block features with the same BCHW shape."""
        return self.body(x) + x


class ScaleAttentionFusion(nn.Module):
    """Fuse same-resolution features produced from multiple image scales."""

    def __init__(self, in_channels, ratio=4):
        """Create channel attention layers for a concatenated multi-scale descriptor."""
        super().__init__()
        hidden_channels = max(in_channels // ratio, 1)
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress1 = nn.Conv2d(in_channels, hidden_channels, 1, 1, 0)
        self.compress2 = nn.Conv2d(hidden_channels, hidden_channels, 1, 1, 0)
        self.excitation = nn.Conv2d(hidden_channels, in_channels, 1, 1, 0)

    def forward(self, x0, x2, x4):
        """Fuse BCHW tensors from original, half, and quarter scales into one BCHW tensor."""
        pooled = torch.cat([self.squeeze(x0), self.squeeze(x2), self.squeeze(x4)], dim=1)
        weights = self.compress1(pooled)
        weights = F.relu(weights)
        weights = self.compress2(weights)
        weights = F.relu(weights)
        weights = torch.sigmoid(self.excitation(weights))
        w0, w2, w4 = torch.chunk(weights, 3, dim=1)
        return x0 * w0 + x2 * w2 + x4 * w4


class StructureAwareMultiscaleBlock(nn.Module):
    """Multi-scale residual block that enriches local features at three resolutions."""

    def __init__(self, in_channel, d_list, inter_num):
        """Create dense blocks for original, half, and quarter resolution feature streams."""
        super().__init__()
        self.original_block = DenseBlock(in_channel, d_list, inter_num)
        self.half_block = DenseBlock(in_channel, d_list, inter_num)
        self.quarter_block = DenseBlock(in_channel, d_list, inter_num)
        self.fusion = ScaleAttentionFusion(3 * in_channel)

    def forward(self, x):
        """Return a residual BCHW tensor after processing `x` at three scales."""
        x0 = x
        x2 = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
        x4 = F.interpolate(x, scale_factor=0.25, mode="bilinear", align_corners=False)
        y0 = self.original_block(x0)
        y2 = self.half_block(x2)
        y4 = self.quarter_block(x4)
        y2 = F.interpolate(y2, size=y0.shape[-2:], mode="bilinear", align_corners=False)
        y4 = F.interpolate(y4, size=y0.shape[-2:], mode="bilinear", align_corners=False)
        return x + self.fusion(y0, y2, y4)
