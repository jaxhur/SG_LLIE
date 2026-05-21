"""Optional CIConv2d structure-prior extraction utility."""

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.image_io import image_to_tensor, load_image
from utils.paths import ensure_dir, list_image_files


EPS = 1e-5


def gaussian_basis_filters(scale, device, k=3):
    """Create Gaussian and derivative basis filters for a learnable scale."""
    std = torch.pow(2, scale)
    filtersize = torch.ceil(k * std + 0.5)
    x = torch.arange(start=-filtersize.item(), end=filtersize.item() + 1, device=device)
    grid_y, grid_x = torch.meshgrid(x, x, indexing="ij")
    g = torch.exp(-((grid_x / std) ** 2) / 2) * torch.exp(-((grid_y / std) ** 2) / 2)
    g = g / torch.sum(g)
    dgdx = -grid_x / (std**3 * 2 * math.pi) * torch.exp(-((grid_x / std) ** 2) / 2) * torch.exp(
        -((grid_y / std) ** 2) / 2
    )
    dgdy = -grid_y / (std**3 * 2 * math.pi) * torch.exp(-((grid_y / std) ** 2) / 2) * torch.exp(
        -((grid_x / std) ** 2) / 2
    )
    dgdx = dgdx / torch.sum(torch.abs(dgdx))
    dgdy = dgdy / torch.sum(torch.abs(dgdy))
    return torch.stack([g, dgdx, dgdy], dim=0)[:, None, :, :]


def invariant_w(e, ex, ey, el, elx, ely, ell, ellx, elly):
    """Return W invariant response from Gaussian color-model derivatives."""
    energy = e + EPS
    wx = ex / energy
    wy = ey / energy
    wlx = elx / energy
    wly = ely / energy
    wllx = ellx / energy
    wlly = elly / energy
    return wx**2 + wy**2 + wlx**2 + wly**2 + wllx**2 + wlly**2


class CIConv2d(nn.Module):
    """Color-invariant convolution module used to extract structure priors."""

    def __init__(self, k=3, scale=0.9):
        """Create Gaussian color-model constants and a learnable scale parameter."""
        super().__init__()
        self.k = k
        self.register_buffer("gcm", torch.tensor([[0.06, 0.63, 0.27], [0.3, 0.04, -0.35], [0.34, -0.6, 0.17]]))
        self.scale = nn.Parameter(torch.tensor([scale]), requires_grad=True)

    def forward(self, batch):
        """Convert BCHW RGB input into a single-channel normalized structure-prior tensor."""
        self.scale.data = torch.clamp(self.scale.data, min=-2.5, max=2.5)
        in_shape = batch.shape
        flat = batch.view(in_shape[0], in_shape[1], -1)
        color = torch.matmul(self.gcm, flat)
        color = color.view(in_shape[0], 3, *in_shape[2:])
        e, el, ell = torch.split(color, 1, dim=1)
        weight = gaussian_basis_filters(scale=self.scale, device=batch.device, k=self.k)
        e_out = F.conv2d(e, weight, padding=int(weight.shape[2] / 2))
        el_out = F.conv2d(el, weight, padding=int(weight.shape[2] / 2))
        ell_out = F.conv2d(ell, weight, padding=int(weight.shape[2] / 2))
        e, ex, ey = torch.split(e_out, 1, dim=1)
        el, elx, ely = torch.split(el_out, 1, dim=1)
        ell, ellx, elly = torch.split(ell_out, 1, dim=1)
        response = invariant_w(e, ex, ey, el, elx, ely, ell, ellx, elly)
        return F.instance_norm(torch.log(response + EPS))


def save_gray(path, tensor):
    """Save a BCHW or CHW single-channel tensor as an 8-bit grayscale image."""
    array = tensor.detach().float().cpu()
    if array.dim() == 4:
        array = array[0, 0]
    elif array.dim() == 3:
        array = array[0]
    array = array.numpy()
    array = (np.clip(array, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    cv2.imwrite(str(path), array)


def extract_priors(input_dir, output_dir, device=None):
    """Extract structure-prior images from `input_dir` into `output_dir`."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = ensure_dir(output_dir)
    model = CIConv2d().to(device).eval()
    with torch.no_grad():
        for image_path in tqdm(list_image_files(input_dir), desc="extract-prior"):
            image = image_to_tensor(load_image(image_path)).unsqueeze(0).to(device)
            prior = model(image)
            save_gray(output_dir / image_path.name, prior)


def parse_args():
    """Parse command-line arguments for standalone structure-prior extraction."""
    parser = argparse.ArgumentParser(description="Extract SG_LLIE structure priors.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device")
    return parser.parse_args()


def main():
    """Run command-line prior extraction."""
    args = parse_args()
    extract_priors(args.input_dir, args.output_dir, device=args.device)


if __name__ == "__main__":
    main()
