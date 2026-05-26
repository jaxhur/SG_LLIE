"""PSNR 和 SSIM 图像质量评价指标。"""

import cv2
import numpy as np
import torch


def _to_numpy_image(image):
    """把 torch Tensor 或 numpy 数组统一转换成 HWC 格式的 numpy 图像。"""
    if isinstance(image, torch.Tensor):
        if image.dim() == 4:
            image = image.squeeze(0)
        image = image.detach().float().cpu().numpy().transpose(1, 2, 0)
    if image.ndim == 2:
        image = image[..., None]
    return image


def calculate_psnr(img1, img2, crop_border=0):
    """计算两张图像的 PSNR。
    输入:
        img1/img2: 待比较图像，支持 Tensor 或 numpy。
        crop_border: 计算前裁掉的边界像素数。
    输出:
        PSNR 数值，越大表示越接近 GT。
    """
    img1 = _to_numpy_image(img1).astype(np.float64)
    img2 = _to_numpy_image(img2).astype(np.float64)
    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes differ: {img1.shape} vs {img2.shape}")
    if crop_border:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
    max_value = 1.0 if max(img1.max(), img2.max()) <= 1.0 else 255.0
    # PSNR核心公式：
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(max_value **2 /mse)


def _ssim_single_channel(img1, img2, max_value):
    """计算单通道图像的 SSIM。"""
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
    """计算多通道图像的平均 SSIM。

    输入:
        img1/img2: 待比较图像，支持 Tensor 或 numpy。
        crop_border: 计算前裁掉的边界像素数。
    输出:
        SSIM 数值，范围通常接近 [0, 1]，越大越相似。
    """
    img1 = _to_numpy_image(img1).astype(np.float64)
    img2 = _to_numpy_image(img2).astype(np.float64)
    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes differ: {img1.shape} vs {img2.shape}")
    if crop_border:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
    max_value = 1.0 if max(img1.max(), img2.max()) <= 1.0 else 255.0
    return float(np.mean([_ssim_single_channel(img1[..., c], img2[..., c], max_value) for c in range(img1.shape[2])]))
