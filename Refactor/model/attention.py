"""SGTB 模块：Structure-Guided Transformer Block。

整体结构对应论文图中的三段残差分支：
    x -> LN -> CSA  -> 残差相加
      -> LN -> SGCA -> 残差相加，结构先验作为 K/V
      -> LN -> FFN  -> 残差相加

本文件所有公开模块都使用 BCHW 格式，避免在主网络中频繁转换维度。
"""

import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_3d(x):
    """把 BCHW 特征展平为 B(HW)C,方便对通道维做 LayerNorm """
    return x.flatten(2).transpose(1, 2).contiguous()


def _to_4d(x, h, w):
    """把 B(HW)C 特征还原为 BCHW 图像特征。"""
    b, _, c = x.shape
    return x.transpose(1, 2).contiguous().view(b, c, h, w)


def _split_heads(x, heads):
    """把 BCHW 特征拆成 B, heads, C_per_head, HW 的多头注意力格式。"""
    b, c, h, w = x.shape
    return x.view(b, heads, c // heads, h * w)


def _merge_heads(x, h, w):
    """把 B, heads, C_per_head, HW 的注意力结果合回 BCHW。"""
    b, heads, c, _ = x.shape
    return x.contiguous().view(b, heads * c, h, w)


class BiasFreeLayerNorm(nn.Module):
    """不带 bias 的 LayerNorm，只学习通道缩放参数。"""

    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        if len(normalized_shape) != 1:
            raise ValueError("LayerNorm expects one channel dimension.")

        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        """在最后一个维度上归一化，输入通常是 B(HW)C。"""
        variance = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(variance + 1e-5) * self.weight

class WithBiasLayerNorm(nn.Module):
    """带 bias 的 LayerNorm，同时学习通道缩放和平移参数。"""

    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        if len(normalized_shape) != 1:
            raise ValueError("LayerNorm expects one channel dimension.")

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        """在最后一个维度上做标准 LayerNorm，输入通常是 B(HW)C。"""
        mean = x.mean(-1, keepdim=True)
        variance = x.var(-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(variance + 1e-5) * self.weight + self.bias

class LayerNorm2d(nn.Module):
    """面向 BCHW 图像特征的 LayerNorm 包装层。"""

    def __init__(self, dim, norm_type="WithBias"):
        super().__init__()
        if norm_type == "BiasFree":
            self.body = BiasFreeLayerNorm(dim)
        else:
            self.body = WithBiasLayerNorm(dim)

    def forward(self, x):
        """先展平空间维，再对通道维归一化，最后恢复 BCHW。"""
        h, w = x.shape[-2:]
        return _to_4d(self.body(_to_3d(x)), h, w)


class ChannelSelfAttention(nn.Module):
    """CSA：通道自注意力，采用QKV的方式，不同通道之间互相交流信息。"""

    def __init__(self, dim, heads=1, bias=False):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}.")

        # 多头注意力的头数
        self.heads = heads
        # 可学习缩放参数，调节注意力分数的尺度。每个 head 都有一个自己的缩放因子。
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        # 1x1 卷积把输入特征从 C 通道映射成 3C 通道,作为qkv
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        # 深度可分离卷积,groups=dim * 3，每个通道单独做 3x3 卷积，不混合通道。
        # 在 Q/K/V 中引入局部空间上下文
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        # 1x1 卷积把注意力结果重新投影一下，输出通道仍然是 C
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        """先用卷积得到 Q/K/V，计算通道和通道之间的注意力关系，最后重新组合特征。"""
        b, _, h, w = x.shape

        # 拆出qkv
        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)

        # 转成多头形式，在每个 head 内做通道维注意力。
        q = _split_heads(q, self.heads)
        k = _split_heads(k, self.heads)
        v = _split_heads(v, self.heads)

        # 沿着空间维 H*W 做 L2 归一化（每个通道被看成长度为 H*W 的向量，归一化）
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # q=[B, heads, C_head, H*W];k.transpose(-2,-1)=[B, heads, H*W, C_head]
        # attention=[B, heads, C_head, C_head]每个通道和其他通道之间的相关性
        attention = (q @ k.transpose(-2, -1)) * self.temperature
        attention = attention.softmax(dim=-1)
        # v=[B, heads, C_head, H*W];out=[B, heads, C_head, H*W]
        out = attention @ v
        out = _merge_heads(out, h, w)
        return self.project_out(out)


class StructureGuidedCrossAttention(nn.Module):
    """SGCA：结构引导交叉注意力，用图像特征产生 Q，结构先验产生 K/V。
        SGCA 和 CSA 的计算形式非常像，本质区别就是 Q/K/V 来自哪里不同。
        CSA中QKV都是来自图像X，而SGCA中Q来自图像X，KV来自结构先验S。
    SGCA 根据图像特征 Q 和结构先验 K 的相关性，从结构先验 V 中聚合出结构引导特征；
    随后在 SGTB 的残差连接中，这个结构引导特征被加到图像增强特征 x 上。
    """

    def __init__(self, dim, heads=1, bias=False):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}.")

        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, 
                                  padding=1, groups=dim, bias=bias)

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(
            dim * 2,
            dim * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, structure):
        """输入图像特征 x 和结构特征 structure，返回结构引导后的图像特征。
            x: [B, C, H, W] 图像特征，作为 Query 的来源。
            structure: [B, C, H, W] 结构先验特征
        """
        b, _, h, w = x.shape

        # 当前图像特征作为 query，结构先验作为 key/value。
        q = self.q_dwconv(self.q(x))
        k, v = self.kv_dwconv(self.kv(structure)).chunk(2, dim=1)

        # 展平成多头通道注意力形式，空间位置作为相关性统计维度。
        q = _split_heads(q, self.heads)
        k = _split_heads(k, self.heads)
        v = _split_heads(v, self.heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attention = (q @ k.transpose(-2, -1)) * self.temperature
        attention = attention.softmax(dim=-1)

        out = attention @ v
        out = _merge_heads(out, h, w)
        return self.project_out(out)


class FeedForwardNetwork(nn.Module):
    """FFN：卷积前馈网络，用于进一步增强局部表达。"""

    def __init__(self, dim, expansion_factor=4, bias=False):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=bias),
            nn.GELU(),
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=hidden_dim,
                bias=bias,
            ),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=bias),
        )

    def forward(self, x):
        """输入输出均为 BCHW，通道数和空间尺寸保持不变。"""
        return self.net(x)


class StructureGuidedTransformerBlock(nn.Module):
    """SGTB：严格按 LN-CSA、LN-SGCA、LN-FFN 三段残差结构组织。"""

    def __init__(self, dim, heads=1, ffn_expansion_factor=4, bias=False, norm_type="WithBias"):
        super().__init__()
        # 原始结构先验通常是 3 通道图，这里投影到当前特征通道数 dim。
        self.structure_proj = nn.Conv2d(3, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.structure_norm = LayerNorm2d(dim, norm_type=norm_type)

        self.norm_csa = LayerNorm2d(dim, norm_type=norm_type)
        self.csa = ChannelSelfAttention(dim=dim, heads=heads, bias=bias)

        self.norm_sgca = LayerNorm2d(dim, norm_type=norm_type)
        self.sgca = StructureGuidedCrossAttention(dim=dim, heads=heads, bias=bias)

        self.norm_ffn = LayerNorm2d(dim, norm_type=norm_type)
        self.ffn = FeedForwardNetwork(dim=dim, expansion_factor=ffn_expansion_factor, bias=bias)

    def forward(self, x, structure):
        """输入图像特征 x 和结构先验 structure，输出结构引导后的同形状特征。"""
        # 调用方通常已经把结构图下采样到同尺度；这里保底对齐空间尺寸。
        if structure.shape[-2:] != x.shape[-2:]:
            structure = F.interpolate(structure, size=x.shape[-2:], mode="bilinear", align_corners=False)

        structure = self.structure_norm(self.structure_proj(structure))

        # 三段子模块都采用 Pre-LN + 残差连接，对应论文图中的三个加号。
        x = x + self.csa(self.norm_csa(x))
        x = x + self.sgca(self.norm_sgca(x), structure)
        x = x + self.ffn(self.norm_ffn(x))
        return x
