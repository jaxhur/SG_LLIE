"""Standalone SG_LLIE inference entrypoint."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from tqdm import tqdm

from metrics.psnr_ssim import calculate_psnr, calculate_ssim
from model import SG_LLIE
from test.self_ensemble import self_ensemble
from utils.checkpoint import load_checkpoint
from utils.config import ConfigLoader, require_keys
from utils.image_io import image_to_tensor, load_image, pad_to_factor, save_image, tensor_to_image
from utils.paths import ensure_dir, paired_by_name


def parse_args():
    """Parse inference CLI arguments, with paths supplied outside YAML."""
    parser = argparse.ArgumentParser(description="Test SG_LLIE without external restoration frameworks.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "sg_llie_ntire25.yaml"))
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--input_s_dir", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--result_dir")
    parser.add_argument("--gt_dir")
    parser.add_argument("--no_self_ensemble", action="store_true")
    return parser.parse_args()


def build_model(config, device):
    """Instantiate SG_LLIE from YAML config and move it to `device`."""
    model_config = dict(config["model"])
    model_config.pop("name", None)
    return SG_LLIE(**model_config).to(device)


def main():
    """Run SG_LLIE inference over all paired input and structure-prior images."""
    args = parse_args()
    require_keys(vars(args), ["input_dir", "input_s_dir", "weights"], "testing paths")
    config = ConfigLoader(args.config).load()
    result_dir = ensure_dir(args.result_dir or config["paths"].get("result_dir", ROOT / "test" / "results"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, device)
    load_checkpoint(args.weights, model, device=device, strict=False)
    model.eval()
    pairs = paired_by_name(args.input_dir, args.input_s_dir)
    gt_pairs = dict((Path(src).name, gt) for src, gt in paired_by_name(args.input_dir, args.gt_dir)) if args.gt_dir else {}
    use_ensemble = config["testing"].get("self_ensemble", True) and not args.no_self_ensemble
    factor = config["testing"].get("factor", 32)
    metric_sums = {"psnr": 0.0, "ssim": 0.0}
    metric_count = 0
    with torch.no_grad():
        for input_path, prior_path in tqdm(pairs, desc="test"):
            image = image_to_tensor(load_image(input_path)).unsqueeze(0).to(device)
            prior = image_to_tensor(load_image(prior_path)).unsqueeze(0).to(device)
            image, (h, w) = pad_to_factor(image, factor)
            prior, _ = pad_to_factor(prior, factor)
            restored = self_ensemble(image, prior, model) if use_ensemble else model(image, prior)[0]
            restored = restored[:, :, :h, :w].clamp(0.0, 1.0)
            save_image(result_dir / Path(input_path).name, tensor_to_image(restored))
            if Path(input_path).name in gt_pairs:
                gt = image_to_tensor(load_image(gt_pairs[Path(input_path).name])).unsqueeze(0)
                metric_sums["psnr"] += calculate_psnr(restored.cpu(), gt)
                metric_sums["ssim"] += calculate_ssim(restored.cpu(), gt)
                metric_count += 1
    if metric_count:
        print(f"PSNR: {metric_sums['psnr'] / metric_count:.4f} SSIM: {metric_sums['ssim'] / metric_count:.4f}")


if __name__ == "__main__":
    main()
