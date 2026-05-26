"""图像读取、保存和 Tensor 转换工具。"""

from pathlib import Path

import cv2
import numpy as np
import torch

from utils.paths import ensure_dir


def load_image(path, color=True):
    """读取图像。

    输入:
        path: 图像路径。
        color: True 时按 RGB 三通道读取；False 时按灰度读取。
    输出:
        float32 numpy 图像，数值范围 [0, 1]。
    """
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
    image = cv2.imread(str(path), flag)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    if color:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = image[..., None]
    return image.astype(np.float32) / 255.0


def save_image(path, image):
    """保存 RGB 或灰度 numpy 图像到 path，自动创建父目录。"""
    path = Path(path)
    ensure_dir(path.parent)
    image = np.clip(image, 0.0, 1.0)
    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor((image * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
    else:
        image = (image.squeeze() * 255.0).round().astype(np.uint8)
    cv2.imwrite(str(path), image)


def image_to_tensor(image):
    """把 HWC numpy 图像转换成 CHW float Tensor。"""
    if image.ndim == 2:
        image = image[..., None]
    return torch.from_numpy(image.transpose(2, 0, 1)).float()


def tensor_to_image(tensor):
    """把 CHW 或 BCHW Tensor 转换成 HWC numpy 图像。"""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    tensor = tensor.detach().float().cpu().clamp(0.0, 1.0)
    array = tensor.numpy().transpose(1, 2, 0)
    return array


def pad_to_factor(x, factor):
    """用反射 padding 把 BCHW 张量的高宽补到 factor 的倍数。"""
    _, _, h, w = x.shape
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    if pad_h == 0 and pad_w == 0:
        return x, (h, w)
    padded = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return padded, (h, w)
