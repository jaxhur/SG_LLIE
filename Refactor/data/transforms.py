"""Synchronized crop, padding, and geometric augmentation utilities."""

import random

import cv2
import numpy as np


def reflect_pad_to_size(images, size):
    """Reflect-pad each HWC image in `images` so height and width are at least `size`."""
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
    """Crop all HWC images at the same random location and return the cropped images."""
    h, w = images[0].shape[:2]
    if h < crop_size or w < crop_size:
        raise ValueError(f"Crop size {crop_size} is larger than image size {(h, w)} after padding.")
    top = random.randint(0, h - crop_size)
    left = random.randint(0, w - crop_size)
    return [image[top : top + crop_size, left : left + crop_size, ...] for image in images]


def augment_geometric(images, enable=True):
    """Apply identical random flip and rotation augmentation to all HWC images."""
    if not enable:
        return images
    mode = random.randint(0, 7)
    return [apply_geometric_mode(image, mode).copy() for image in images]


def apply_geometric_mode(image, mode):
    """Apply one of eight flip/rotation modes to HWC image `image` and return it."""
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
