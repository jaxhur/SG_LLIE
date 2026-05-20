"""PSNR and SSIM image quality metrics."""

import cv2
import numpy as np
import torch


def _to_numpy_image(image):
    """Convert tensor or numpy input to HWC numpy array in `[0, 1]` or `[0, 255]`."""
    if isinstance(image, torch.Tensor):
        if image.dim() == 4:
            image = image.squeeze(0)
        image = image.detach().float().cpu().numpy().transpose(1, 2, 0)
    if image.ndim == 2:
        image = image[..., None]
    return image


def calculate_psnr(img1, img2, crop_border=0):
    """Calculate PSNR between two images with matching shapes."""
    img1 = _to_numpy_image(img1).astype(np.float64)
    img2 = _to_numpy_image(img2).astype(np.float64)
    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes differ: {img1.shape} vs {img2.shape}")
    if crop_border:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
    max_value = 1.0 if max(img1.max(), img2.max()) <= 1.0 else 255.0
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(max_value / np.sqrt(mse))


def _ssim_single_channel(img1, img2, max_value):
    """Calculate SSIM for one single-channel image pair."""
    c1 = (0.01 * max_value) ** 2
    c2 = (0.03 * max_value) ** 2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()


def calculate_ssim(img1, img2, crop_border=0):
    """Calculate channel-averaged SSIM between two images with matching shapes."""
    img1 = _to_numpy_image(img1).astype(np.float64)
    img2 = _to_numpy_image(img2).astype(np.float64)
    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes differ: {img1.shape} vs {img2.shape}")
    if crop_border:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
    max_value = 1.0 if max(img1.max(), img2.max()) <= 1.0 else 255.0
    return float(np.mean([_ssim_single_channel(img1[..., c], img2[..., c], max_value) for c in range(img1.shape[2])]))
