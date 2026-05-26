"""数据增强工具。

这里的增强都要求 low、gt、low_s 等图像同步变化，
否则输入和标签会错位，训练会被破坏。
"""

import random

import cv2
import numpy as np


def reflect_pad_to_size(images, size):
    """把每张 HWC 图像用反射 padding 补到至少 size x size。

    输入:
        images: 多张 HWC 图像列表。
        size: 目标最小边长。
    输出:
        padding 后的图像列表。
    """
    padded = []
    for image in images:
        h, w = image.shape[:2]
        pad_h = max(0, size - h)
        pad_w = max(0, size - w)
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        padded.append(image)
    return padded


def paired_random_crop(images, crop_size):
    """对多张配对图像执行同位置随机裁剪。

    输入:
        images: 已经空间对齐的图像列表。
        crop_size: 裁剪出的 patch 大小。
    输出:
        同一位置裁剪得到的图像列表。
    """
    h, w = images[0].shape[:2]
    if h < crop_size or w < crop_size:
        raise ValueError(f"Crop size {crop_size} is larger than image size {(h, w)} after padding.")
    top = random.randint(0, h - crop_size)
    left = random.randint(0, w - crop_size)
    return [image[top : top + crop_size, left : left + crop_size, ...] for image in images]


def augment_geometric(images, enable=True):
    """对多张配对图像执行相同的随机翻转/旋转增强。"""
    if not enable:
        return images
    mode = random.randint(0, 7)
    return [apply_geometric_mode(image, mode).copy() for image in images]


def apply_geometric_mode(image, mode):
    """根据 mode 对单张 HWC 图像执行指定几何变换。

    mode 范围为 0 到 7，覆盖原图、上下翻转、90/180/270 度旋转及组合。
    """
    if mode == 0:
        return image
    if mode == 1:
        return np.flipud(image)
    if mode == 2:
        return np.rot90(image)
    if mode == 3:
        return np.flipud(np.rot90(image))
    if mode == 4:
        return np.rot90(image, k=2)
    if mode == 5:
        return np.flipud(np.rot90(image, k=2))
    if mode == 6:
        return np.rot90(image, k=3)
    if mode == 7:
        return np.flipud(np.rot90(image, k=3))
    raise ValueError(f"Invalid geometric augmentation mode: {mode}")
