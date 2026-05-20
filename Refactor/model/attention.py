"""Attention and normalization layers used by SG_LLIE."""

import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class GELU(nn.Module):
    """Apply GELU activation to the input tensor and return the activated tensor."""

    def forward(self, x):
        """Return `x` after GELU activation; input and output shapes are identical."""
        return F.gelu(x)


class PreNorm(nn.Module):
    """Normalize channel-last features before passing them to a wrapped module."""

    def __init__(self, dim, fn):
        """Create a layer norm over `dim` channels and store the callable module `fn`."""
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        """Normalize channel-last tensor `x`, call `fn`, and return its output."""
        return self.fn(self.norm(x), *args, **kwargs)


class FeedForward(nn.Module):
    """Depth-wise convolutional feed-forward block for channel-last features."""

    def __init__(self, dim, mult=4):
        """Build a convolutional MLP that maps `dim` channels back to `dim` channels."""
        super().__init__()
        hidden_dim = dim * mult
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=False, groups=hidden_dim),
            GELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """Accept a BHWC tensor and return a BHWC tensor with the same shape."""
        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        return out.permute(0, 2, 3, 1)


def to_3d(x):
    """Flatten a BCHW tensor into B(HW)C for layer normalization."""
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    """Restore a B(HW)C tensor into BCHW using spatial size `h` by `w`."""
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFreeLayerNorm(nn.Module):
    """Layer normalization without learnable bias for flattened image features."""

    def __init__(self, normalized_shape):
        """Store a learnable scale for one channel dimension."""
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        if len(normalized_shape) != 1:
            raise ValueError("Layer norm expects a single channel dimension.")
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        """Normalize the last dimension of `x` and return the scaled tensor."""
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):
    """Layer normalization with learnable scale and bias for image features."""

    def __init__(self, normalized_shape):
        """Create learnable scale and bias parameters for one channel dimension."""
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        if len(normalized_shape) != 1:
            raise ValueError("Layer norm expects a single channel dimension.")
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        """Normalize the last dimension of `x` and return the affine-transformed tensor."""
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm2d(nn.Module):
    """Apply channel-wise layer normalization to BCHW image tensors."""

    def __init__(self, dim, norm_type="WithBias"):
        """Choose bias-free or bias-enabled normalization for `dim` channels."""
        super().__init__()
        if norm_type == "BiasFree":
            self.body = BiasFreeLayerNorm(dim)
        else:
            self.body = WithBiasLayerNorm(dim)

    def forward(self, x):
        """Normalize BCHW tensor `x` over channels and return a BCHW tensor."""
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class IlluminationGuidedMSA(nn.Module):
    """Multi-head self-attention used inside SG_LLIE feature blocks."""

    def __init__(self, dim, dim_head=64, heads=8):
        """Create query/key/value projections for `heads` attention heads."""
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x):
        """Accept a BHWC tensor and return self-attended BHWC features."""
        b, h, w, c = x.shape
        tokens = x.reshape(b, h * w, c)
        q_inp = self.to_q(tokens)
        k_inp = self.to_k(tokens)
        v_inp = self.to_v(tokens)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads),
            (q_inp, k_inp, v_inp),
        )
        q = F.normalize(q.transpose(-2, -1), dim=-1, p=2)
        k = F.normalize(k.transpose(-2, -1), dim=-1, p=2)
        v = v.transpose(-2, -1)
        attn = (k @ q.transpose(-2, -1)) * self.rescale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.permute(0, 3, 1, 2).reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(out).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return out_c + out_p


class CrossAttention(nn.Module):
    """Inject structure-prior features into restoration features through attention."""

    def __init__(self, dim, num_heads, bias):
        """Create query projection for image features and key/value projection for prior features."""
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, 3, 1, 1, groups=dim * 2, bias=bias)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, s):
        """Use BCHW feature tensor `x` and BCHW prior tensor `s`; return BHWC features."""
        b, _, h, w = x.shape
        kv = self.kv_dwconv(self.kv(s))
        k, v = kv.chunk(2, dim=1)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        q = self.q_dwconv(self.q(x))
        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out.permute(0, 2, 3, 1)


class IlluminationGuidedAttentionBlock(nn.Module):
    """Attention block that combines image self-attention and structure-prior cross-attention."""

    def __init__(self, dim, dim_head=64, heads=8, num_blocks=2):
        """Build `num_blocks` attention layers for `dim`-channel BCHW features."""
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                nn.ModuleList(
                    [
                        IlluminationGuidedMSA(dim=dim, dim_head=dim_head, heads=heads),
                        LayerNorm2d(dim),
                        CrossAttention(dim, num_heads=heads, bias=False),
                        PreNorm(dim, FeedForward(dim=dim)),
                    ]
                )
            )
        self.s_conv = nn.Conv2d(3, dim, kernel_size=3, stride=1, padding=1)
        self.s_norm = LayerNorm2d(dim)

    def forward(self, x, s):
        """Fuse BCHW feature tensor `x` with BCHW structure prior `s` and return BCHW features."""
        s = self.s_norm(self.s_conv(s))
        x = x.permute(0, 2, 3, 1)
        for attn, cross_norm, cross_attn, ff in self.blocks:
            x = attn(x) + x
            x = cross_attn(cross_norm(x.permute(0, 3, 1, 2)), s) + x
            x = ff(x) + x
        return x.permute(0, 3, 1, 2)
