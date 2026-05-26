"""SG_LLIE 可复用基础模块。

这里放的是卷积块、残差密集块、多尺度融合块等通用组件。
这些模块被编码器和解码器复用，用来减少主网络文件的复杂度。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """只包含一层卷积的基础模块，不附带激活函数。"""

    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, 
                 padding=0, stride=1):
        """初始化卷积层。

        输入参数:
            in_channel: 输入通道数。
            out_channel: 输出通道数。
            kernel_size: 卷积核大小。
            dilation_rate: 空洞卷积膨胀率。
            padding: 填充大小。
            stride: 步长。
        输出:
            无返回值。
        """
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
        """对输入特征 x 执行卷积，输入输出均为 [B, C, H, W] 格式。"""
        return self.conv(x)


class ConvReLUBlock(nn.Module):
    """卷积 + ReLU 激活模块。"""

    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        """初始化卷积和 ReLU。

        输入参数含义与 ConvBlock 相同。
        输出:
            无返回值。
        """
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
        """对输入 x 执行卷积和 ReLU，返回激活后的特征。"""
        return self.conv(x)


class DenseBlock(nn.Module):
    """密集连接的空洞卷积块，不包含最外层残差相加。"""

    def __init__(self, in_channel, d_list, inter_num):
        """初始化密集块。

        输入参数:
            in_channel: 输入和最终输出通道数。
            d_list: 每个卷积层使用的 dilation 列表。
            inter_num: 每个中间卷积层输出通道数。
        作用:
            每一层都会把新特征和已有特征拼接，从而增强局部表达能力。
        """
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
        """输入 x，经过多层密集连接卷积，返回和 x 通道数一致的特征。"""
        features = x
        for conv_layer in self.conv_layers:
            new_features = conv_layer(features)
            features = torch.cat([new_features, features], dim=1)
        return self.conv_post(features)


class ResidualDenseBlock(nn.Module):
    """带残差连接的密集空洞卷积块。"""

    def __init__(self, in_channel, d_list, inter_num):
        """初始化残差密集块，内部使用 DenseBlock。"""
        super().__init__()
        self.body = DenseBlock(in_channel, d_list, inter_num)

    def forward(self, x):
        """返回 DenseBlock(x) + x，保持输入输出形状一致。"""
        return self.body(x) + x


class ScaleAttentionFusion(nn.Module):
    """多尺度特征通道注意力融合模块。"""

    def __init__(self, in_channels, ratio=4):
        """初始化融合层。

        输入参数:
            in_channels: 三个尺度特征拼接后的通道数。
            ratio: 通道压缩比例。
        作用:
            根据全局平均池化得到的描述子，为三个尺度分配自适应权重。
        """
        super().__init__()
        hidden_channels = max(in_channels // ratio, 1)
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化，输出 [B, C, 1, 1] 的描述子
        self.compress1 = nn.Conv2d(in_channels, hidden_channels, 1, 1, 0) # 1x1卷积
        self.compress2 = nn.Conv2d(hidden_channels, hidden_channels, 1, 1, 0) # 1x1卷积
        self.excitation = nn.Conv2d(hidden_channels, in_channels, 1, 1, 0) # 1x1卷积，输出每个尺度的权重

    def forward(self, x0, x2, x4):
        """融合原尺度、1/2 尺度、1/4 尺度特征，输出融合后的 BCHW 特征。"""
        pooled = torch.cat([self.squeeze(x0), self.squeeze(x2),
                             self.squeeze(x4)], dim=1) 
        weights = self.compress1(pooled)
        weights = F.relu(weights)
        weights = self.compress2(weights)
        weights = F.relu(weights)
        weights = torch.sigmoid(self.excitation(weights)) # 输出范围在 (0, 1)，表示每个尺度的权重
        w0, w2, w4 = torch.chunk(weights, 3, dim=1) # 将权重分成三个部分，分别对应三个尺度
        return x0 * w0 + x2 * w2 + x4 * w4


class StructureAwareMultiscaleBlock(nn.Module):
    # FIXME 实际上是Semantic-Aligned Scale-Aware Module语义对齐的尺度感知模块,判断当前图像更需要哪个尺度的信息

    """结构感知多尺度块，在三个分辨率上提取并融合局部特征。"""

    def __init__(self, in_channel, d_list, inter_num):
        """初始化三个尺度的 DenseBlock 和融合模块。"""
        super().__init__()
        self.original_block = DenseBlock(in_channel, d_list, inter_num)
        self.half_block = DenseBlock(in_channel, d_list, inter_num)
        self.quarter_block = DenseBlock(in_channel, d_list, inter_num)
        self.fusion = ScaleAttentionFusion(3 * in_channel)

    def forward(self, x):
        """输入特征 x，分别在原尺度、半尺度、四分之一尺度处理后融合，并残差返回。"""
        x0 = x
        # 双线性下采样到一半大小、四分之一大小
        x2 = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
        x4 = F.interpolate(x, scale_factor=0.25, mode="bilinear", align_corners=False)
        y0 = self.original_block(x0) 
        y2 = self.half_block(x2) 
        y4 = self.quarter_block(x4)
        # 双线性上采样，size目标空间尺寸
        y2 = F.interpolate(y2, size=y0.shape[-2:], mode="bilinear", align_corners=False)
        y4 = F.interpolate(y4, size=y0.shape[-2:], mode="bilinear", align_corners=False)
        return x + self.fusion(y0, y2, y4)
