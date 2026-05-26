"""SG_LLIE 图像增强网络。

这个文件定义模型的主体结构：编码器、解码器以及每一级特征处理模块。
模型输入是低照度图像 x 和对应结构先验 s，输出三个尺度的增强结果。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention import StructureGuidedTransformerBlock
from model.modules import ConvBlock, ConvReLUBlock, ResidualDenseBlock, StructureAwareMultiscaleBlock


class SG_LLIE(nn.Module):
    """结构先验引导的低照度图像的整体结构增强网络。"""

    def __init__(
        self,
        en_feature_num=48,
        en_inter_num=32,
        de_feature_num=64,
        de_inter_num=32,
        sam_number=2,
    ):
        """初始化模型。

        输入参数:
            en_feature_num: 编码器基础通道数。
            en_inter_num: 编码器密集块内部通道数。
            de_feature_num: 解码器基础通道数。
            de_inter_num: 解码器密集块内部通道数。
            sam_number: 每一级使用的多尺度结构感知模块数量。
        输出:
            无返回值，构建编码器和解码器子模块。
        """
        super().__init__()
        self.encoder = Encoder(
            feature_num=en_feature_num,
            inter_num=en_inter_num,
            sam_number=sam_number,
        )
        self.decoder = Decoder(
            en_num=en_feature_num,
            feature_num=de_feature_num,
            inter_num=de_inter_num,
            sam_number=sam_number,
        )

    def forward(self, x, s):
        """前向传播。

        输入:
            x: 低照度图像张量，形状为 [B, 3, H, W]，数值范围通常为 [0, 1]。
            s: 结构先验张量，形状为 [B, 3, H, W]，需要和 x 空间对齐。
        输出:
            out1: 原图尺度增强结果，形状 [B, 3, H, W]。
            out2: 1/2 尺度增强结果，用于多尺度监督。
            out3: 1/4 尺度增强结果，用于多尺度监督。
        作用:
            自动把输入 padding 到 32 的倍数，经过编码器和解码器后再裁回原尺寸。
        """
        _, _, h, w = x.shape
        # 输入高宽需要补齐到的倍数，后续编码器和解码器中会有下采样
        rate = 32
        pad_h = (rate - h % rate) % rate
        pad_w = (rate - w % rate) % rate
        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
            s = F.pad(s, (0, pad_w, 0, pad_h), mode="reflect")
        y1, y2, y3 = self.encoder(x, s)
        # TODO 这里是否可以加上中间层
        # NOTE 不太懂啊
        out1, out2, out3 = self.decoder(y1, y2, y3, s)
        out1 = out1[:, :, :h, :w]
        out2 = out2[:, :, : h // 2, : w // 2]
        out3 = out3[:, :, : h // 4, : w // 4]
        return out1, out2, out3


class Encoder(nn.Module):
    """三层编码器，用结构先验引导低照度图像特征提取。"""

    def __init__(self, feature_num, inter_num, sam_number):
        """初始化编码器。

        输入参数:
            feature_num: 第一层编码特征通道数。
            inter_num: 密集块内部通道数。
            sam_number: 每一级多尺度结构感知模块数量。
        输出:
            无返回值，构建 stem 和 3 个编码层。
        """
        super().__init__()
        self.conv_first = nn.Sequential(
            nn.Conv2d(
                12,
                feature_num,
                kernel_size=5,
                stride=1,
                padding=2,
                bias=True,
            ),
            nn.ReLU(inplace=True),
        )
        self.encoder_1 = EncoderLevel(feature_num, inter_num, level=1, sam_number=sam_number)
        self.encoder_2 = EncoderLevel(2 * feature_num, inter_num, level=2, sam_number=sam_number)
        self.encoder_3 = EncoderLevel(4 * feature_num, inter_num, level=3, sam_number=sam_number)

    def forward(self, x, s):
        """编码输入图像。

        输入:
            x: 低照度图像张量 [B, 3, H, W]。
            s: 结构先验张量 [B, 3, H, W]。
        输出:
            out1/out2/out3: 三个层级的编码特征，分辨率逐级降低、通道逐级增加。
        """
        # [B, 3, H, W] -> [B, 12, H/2, W/2]
        x = F.pixel_unshuffle(x, 2)
        # [B, 12, H/2, W/2]->[B, feature_num=48, H/2, W/2]
        # TRICK: 使用5x5卷积下采样后的特征上引入空间上下文
        x = self.conv_first(x)
        # 第1层encoder: out1=[B, 48, H/2, W/2],down1=[B, 96, H/4, W/4]
        out1, down1 = self.encoder_1(x, s)
        # 第2层encoder: out2=[B, 96, H/4, W/4],down2=[B, 192, H/8, W/8]
        out2, down2 = self.encoder_2(down1, s)
        # 第3层encoder: out3=[B, 192, H/8, W/8]
        out3 = self.encoder_3(down2, s)
        return out1, out2, out3


class EncoderLevel(nn.Module):
    """单个encoder层，包含残差密集块、结构注意力、多尺度模块和可选下采样。"""

    def __init__(self, feature_num, inter_num, level, sam_number):
        """初始化一个编码层。

        输入参数:
            feature_num: 当前层输入和输出的特征通道数。
            inter_num: 密集块内部通道数。
            level: 当前层级编号，用来决定结构先验下采样比例。
            sam_number: 多尺度结构感知模块数量。
        输出:
            无返回值。
        """
        super().__init__()
        self.rdb = ResidualDenseBlock(feature_num, (1, 2, 1), inter_num)
        self.sgtb = StructureGuidedTransformerBlock(dim=feature_num, heads=1)
        self.multiscale_blocks = nn.ModuleList(
            [StructureAwareMultiscaleBlock(feature_num, (1, 2, 3, 2, 1), inter_num)
             for _ in range(sam_number)]
        )
        self.level = level
        if level < 3:
            self.down = nn.Sequential(
                nn.Conv2d(feature_num, 2 * feature_num, kernel_size=3, stride=2, padding=1, bias=True),
                nn.ReLU(inplace=True),
            )

    def forward(self, x, s):
        """处理当前层特征。

        输入:
            x: 当前层图像特征 [B, C, H, W]。
            s: 原始结构先验 [B, 3, H0, W0]。
        输出:
            level < 3 时返回 (out, down)，分别是当前层输出和下采样后特征；
            level == 3 时只返回 out。
        """
        # DRDB: 残差密集块,输入输出通道数都等于 feature_num，空间尺寸不变
        out = self.rdb(x)
        scale_factor = 1 / (2**self.level)
        s_level = F.interpolate(s, scale_factor=scale_factor, mode="bilinear", align_corners=False)
        # SGTB: 结构引导注意力块，输入输出通道数都等于 feature_num，空间尺寸不变
        out = self.sgtb(out, s_level)
        for block in self.multiscale_blocks:
            out = block(out)
        if self.level < 3:
            return out, self.down(out)
        return out


class Decoder(nn.Module):
    """三层解码器，逐级融合编码特征并恢复 RGB 图像。"""

    def __init__(self, en_num, feature_num, inter_num, sam_number):
        """初始化解码器。

        输入参数:
            en_num: 编码器基础通道数，用于计算 skip 特征通道。
            feature_num: 解码器特征通道数。
            inter_num: 解码器密集块内部通道数。
            sam_number: 每一级多尺度结构感知模块数量。
        输出:
            无返回值。
        """
        super().__init__()
        self.preconv_3 = ConvReLUBlock(4 * en_num, feature_num, 3, padding=1)
        self.decoder_3 = DecoderLevel(feature_num, inter_num, sam_number, level=3)
        self.preconv_2 = ConvReLUBlock(2 * en_num + feature_num, feature_num, 3, padding=1)
        self.decoder_2 = DecoderLevel(feature_num, inter_num, sam_number, level=2)
        self.preconv_1 = ConvReLUBlock(en_num + feature_num, feature_num, 3, padding=1)
        self.decoder_1 = DecoderLevel(feature_num, inter_num, sam_number, level=1)

    def forward(self, y1, y2, y3, s):
        """解码三个层级的特征。

        输入:
            y1/y2/y3: 编码器输出的三个层级特征。
            s: 原始结构先验。
        输出:
            out1/out2/out3: 原尺度、1/2 尺度、1/4 尺度增强图。
        """
        x3 = self.preconv_3(y3)
        out3, feat3 = self.decoder_3(x3, s)
        x2 = self.preconv_2(torch.cat([y2, feat3], dim=1))
        out2, feat2 = self.decoder_2(x2, s)
        x1 = self.preconv_1(torch.cat([y1, feat2], dim=1))
        out1 = self.decoder_1(x1, s, return_feature=False)
        return out1, out2, out3


class DecoderLevel(nn.Module):
    """单个解码层，预测当前尺度 RGB 图像，并可输出给上一层使用的特征。"""

    def __init__(self, feature_num, inter_num, sam_number, level):
        """初始化一个解码层。

        输入参数:
            feature_num: 当前解码特征通道数。
            inter_num: 密集块内部通道数。
            sam_number: 多尺度结构感知模块数量。
            level: 当前层级编号，用于调整结构先验尺度。
        输出:
            无返回值。
        """
        super().__init__()
        self.rdb = ResidualDenseBlock(feature_num, (1, 2, 1), inter_num)
        self.sgtb = StructureGuidedTransformerBlock(dim=feature_num, heads=1)
        self.multiscale_blocks = nn.ModuleList(
            [StructureAwareMultiscaleBlock(feature_num, (1, 2, 3, 2, 1), inter_num)
             for _ in range(sam_number)]
        )
        self.conv = ConvBlock(feature_num, 12, kernel_size=3, padding=1)
        self.level = level

    def forward(self, x, s, return_feature=True):
        """执行当前层解码。

        输入:
            x: 当前层输入特征 [B, C, H, W]。
            s: 原始结构先验。
            return_feature: 是否额外返回上采样特征，供下一层解码器融合。
        输出:
            return_feature=True 时返回 (rgb, feature)；
            return_feature=False 时只返回 rgb。
        """
        out = self.rdb(x)
        scale_factor = 1 / (2**self.level)
        s_level = F.interpolate(s, scale_factor=scale_factor, mode="bilinear", align_corners=False)
        out = self.sgtb(out, s_level)
        for block in self.multiscale_blocks:
            out = block(out)
        rgb = F.pixel_shuffle(self.conv(out), 2)
        if return_feature:
            feature = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
            return rgb, feature
        return rgb
