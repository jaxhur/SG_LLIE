"""Image loading, saving, and tensor conversion utilities."""

from pathlib import Path

import cv2
import numpy as np
import torch

from utils.paths import ensure_dir


def load_image(path, color=True):
    """Load an image as RGB float32 in `[0, 1]`; grayscale files are expanded to 3 channels."""
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
    """Save RGB or grayscale numpy image to `path`, creating parent folders first."""
    path = Path(path)
    ensure_dir(path.parent)
    image = np.clip(image, 0.0, 1.0)
    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor((image * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
    else:
        image = (image.squeeze() * 255.0).round().astype(np.uint8)
    cv2.imwrite(str(path), image)


def image_to_tensor(image):
    """Convert HWC numpy image in `[0, 1]` to CHW float tensor."""
    if image.ndim == 2:
        image = image[..., None]
    return torch.from_numpy(image.transpose(2, 0, 1)).float()


def tensor_to_image(tensor):
    """Convert CHW or BCHW tensor in `[0, 1]` to HWC RGB numpy image."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    tensor = tensor.detach().float().cpu().clamp(0.0, 1.0)
    array = tensor.numpy().transpose(1, 2, 0)
    return array


def pad_to_factor(x, factor):
    """Reflect-pad BCHW tensor `x` so height and width are divisible by `factor`."""
    _, _, h, w = x.shape
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    if pad_h == 0 and pad_w == 0:
        return x, (h, w)
    padded = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return padded, (h, w)
