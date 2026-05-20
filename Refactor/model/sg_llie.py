"""SG_LLIE restoration network."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention import IlluminationGuidedAttentionBlock
from model.modules import ConvBlock, ConvReLUBlock, ResidualDenseBlock, StructureAwareMultiscaleBlock


class SG_LLIE(nn.Module):
    """Structure-guided low-light image enhancement network."""

    def __init__(
        self,
        en_feature_num=48,
        en_inter_num=32,
        de_feature_num=64,
        de_inter_num=32,
        sam_number=2,
    ):
        """Create encoder and decoder modules with the requested channel sizes."""
        super().__init__()
        self.encoder = Encoder(feature_num=en_feature_num, inter_num=en_inter_num, sam_number=sam_number)
        self.decoder = Decoder(
            en_num=en_feature_num,
            feature_num=de_feature_num,
            inter_num=de_inter_num,
            sam_number=sam_number,
        )

    def forward(self, x, s):
        """Enhance low-light tensor `x` using structure prior `s`; return three RGB output scales."""
        _, _, h, w = x.shape
        rate = 32
        pad_h = (rate - h % rate) % rate
        pad_w = (rate - w % rate) % rate
        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
            s = F.pad(s, (0, pad_w, 0, pad_h), mode="reflect")
        y1, y2, y3 = self.encoder(x, s)
        out1, out2, out3 = self.decoder(y1, y2, y3, s)
        out1 = out1[:, :, :h, :w]
        out2 = out2[:, :, : h // 2, : w // 2]
        out3 = out3[:, :, : h // 4, : w // 4]
        return out1, out2, out3


class Encoder(nn.Module):
    """Three-level encoder that extracts image features guided by structure priors."""

    def __init__(self, feature_num, inter_num, sam_number):
        """Build the pixel-unshuffle stem and three encoder levels."""
        super().__init__()
        self.conv_first = nn.Sequential(
            nn.Conv2d(12, feature_num, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
        )
        self.encoder_1 = EncoderLevel(feature_num, inter_num, level=1, sam_number=sam_number)
        self.encoder_2 = EncoderLevel(2 * feature_num, inter_num, level=2, sam_number=sam_number)
        self.encoder_3 = EncoderLevel(4 * feature_num, inter_num, level=3, sam_number=sam_number)

    def forward(self, x, s):
        """Encode BCHW image tensor `x` and prior `s`; return three feature maps."""
        x = F.pixel_unshuffle(x, 2)
        x = self.conv_first(x)
        out1, down1 = self.encoder_1(x, s)
        out2, down2 = self.encoder_2(down1, s)
        out3 = self.encoder_3(down2, s)
        return out1, out2, out3


class EncoderLevel(nn.Module):
    """One encoder level with residual dense, attention, multi-scale, and optional downsample blocks."""

    def __init__(self, feature_num, inter_num, level, sam_number):
        """Create the level-specific feature processors and downsampler."""
        super().__init__()
        self.rdb = ResidualDenseBlock(feature_num, (1, 2, 1), inter_num)
        self.attention = IlluminationGuidedAttentionBlock(dim=feature_num, dim_head=feature_num, heads=1)
        self.multiscale_blocks = nn.ModuleList(
            [StructureAwareMultiscaleBlock(feature_num, (1, 2, 3, 2, 1), inter_num) for _ in range(sam_number)]
        )
        self.level = level
        if level < 3:
            self.down = nn.Sequential(
                nn.Conv2d(feature_num, 2 * feature_num, kernel_size=3, stride=2, padding=1, bias=True),
                nn.ReLU(inplace=True),
            )

    def forward(self, x, s):
        """Process BCHW features `x` with prior `s`; return level output and optional downsampled output."""
        out = self.rdb(x)
        scale_factor = 1 / (2**self.level)
        s_level = F.interpolate(s, scale_factor=scale_factor, mode="bilinear", align_corners=False)
        out = self.attention(out, s_level)
        for block in self.multiscale_blocks:
            out = block(out)
        if self.level < 3:
            return out, self.down(out)
        return out


class Decoder(nn.Module):
    """Three-level decoder that reconstructs RGB outputs at multiple scales."""

    def __init__(self, en_num, feature_num, inter_num, sam_number):
        """Build decoder levels and skip-connection preprocessing convolutions."""
        super().__init__()
        self.preconv_3 = ConvReLUBlock(4 * en_num, feature_num, 3, padding=1)
        self.decoder_3 = DecoderLevel(feature_num, inter_num, sam_number, level=3)
        self.preconv_2 = ConvReLUBlock(2 * en_num + feature_num, feature_num, 3, padding=1)
        self.decoder_2 = DecoderLevel(feature_num, inter_num, sam_number, level=2)
        self.preconv_1 = ConvReLUBlock(en_num + feature_num, feature_num, 3, padding=1)
        self.decoder_1 = DecoderLevel(feature_num, inter_num, sam_number, level=1)

    def forward(self, y1, y2, y3, s):
        """Decode three encoder feature maps with prior `s`; return full, half, and quarter outputs."""
        x3 = self.preconv_3(y3)
        out3, feat3 = self.decoder_3(x3, s)
        x2 = self.preconv_2(torch.cat([y2, feat3], dim=1))
        out2, feat2 = self.decoder_2(x2, s)
        x1 = self.preconv_1(torch.cat([y1, feat2], dim=1))
        out1 = self.decoder_1(x1, s, return_feature=False)
        return out1, out2, out3


class DecoderLevel(nn.Module):
    """One decoder level that predicts an RGB output and optionally an upsampled feature map."""

    def __init__(self, feature_num, inter_num, sam_number, level):
        """Create residual, attention, multi-scale, and RGB projection layers for this level."""
        super().__init__()
        self.rdb = ResidualDenseBlock(feature_num, (1, 2, 1), inter_num)
        self.attention = IlluminationGuidedAttentionBlock(dim=feature_num, dim_head=feature_num, heads=1)
        self.multiscale_blocks = nn.ModuleList(
            [StructureAwareMultiscaleBlock(feature_num, (1, 2, 3, 2, 1), inter_num) for _ in range(sam_number)]
        )
        self.conv = ConvBlock(feature_num, 12, kernel_size=3, padding=1)
        self.level = level

    def forward(self, x, s, return_feature=True):
        """Return an RGB output and, when requested, a feature map for the next decoder level."""
        out = self.rdb(x)
        scale_factor = 1 / (2**self.level)
        s_level = F.interpolate(s, scale_factor=scale_factor, mode="bilinear", align_corners=False)
        out = self.attention(out, s_level)
        for block in self.multiscale_blocks:
            out = block(out)
        rgb = F.pixel_shuffle(self.conv(out), 2)
        if return_feature:
            feature = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
            return rgb, feature
        return rgb
