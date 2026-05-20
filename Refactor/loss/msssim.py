"""Differentiable SSIM and MS-SSIM losses for PyTorch tensors."""

import torch
import torch.nn.functional as F


def gaussian(window_size, sigma, device, dtype):
    """Return a normalized 1D Gaussian vector on `device` with `dtype`."""
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def create_window(window_size, channel, device, dtype):
    """Create a depth-wise 2D Gaussian window for `channel` channels."""
    one_d = gaussian(window_size, 1.5, device, dtype).unsqueeze(1)
    two_d = one_d @ one_d.t()
    return two_d.expand(channel, 1, window_size, window_size).contiguous()


def ssim(pred, target, window_size=11, size_average=True, val_range=1.0):
    """Compute SSIM and contrast sensitivity between BCHW tensors in `[0, val_range]`."""
    _, channel, _, _ = pred.size()
    window = create_window(window_size, channel, pred.device, pred.dtype)
    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=channel) - mu1_mu2
    c1 = (0.01 * val_range) ** 2
    c2 = (0.03 * val_range) ** 2
    cs = (2.0 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2.0 * mu1_mu2 + c1) * cs) / (mu1_sq + mu2_sq + c1)
    if size_average:
        return ssim_map.mean(), cs.mean()
    return ssim_map.mean([1, 2, 3]), cs.mean([1, 2, 3])


def ms_ssim(pred, target, window_size=11, size_average=True, normalize=True):
    """Compute multi-scale SSIM between BCHW tensors and return a similarity score."""
    weights = pred.new_tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    levels = min(len(weights), max(1, int(torch.log2(torch.tensor(min(pred.shape[-2:]), device=pred.device)).item()) - 1))
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(pred, target, window_size=window_size, size_average=size_average)
        if normalize:
            sim = (sim + 1.0) / 2.0
            cs = (cs + 1.0) / 2.0
        mssim.append(sim)
        mcs.append(cs)
        if min(pred.shape[-2:]) < 2:
            break
        pred = F.avg_pool2d(pred, kernel_size=2, stride=2)
        target = F.avg_pool2d(target, kernel_size=2, stride=2)
    weights = weights[: len(mssim)]
    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)
    if len(mssim) == 1:
        return mssim[0]
    return torch.prod((mcs[:-1] ** weights[:-1]) * (mssim[-1] ** weights[-1]))
