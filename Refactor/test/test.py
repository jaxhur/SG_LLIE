"""SG_LLIE 独立测试/推理入口。

这个脚本用于加载训练好的 checkpoint，对测试集图片进行增强，
并在提供 GT 时计算 PSNR/SSIM。
"""

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
    """解析测试命令行参数。

    输入:
        无显式输入，读取命令行。
    输出:
        argparse.Namespace，包含输入目录、结构先验目录、权重路径、输出目录等。
    """
    parser = argparse.ArgumentParser(description="Test SG_LLIE without external restoration frameworks.")
    # 默认使用 Refactor/configs 下的测试配置；也可以通过命令行换成别的 YAML。
    parser.add_argument("--config",default=str(ROOT / "configs" / "sg_llie_ntire25.yaml"))
    # input_dir 是待增强低照度图像目录，input_s_dir 是与之同名配对的结构先验目录。
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--input_s_dir", required=True)
    # weights 指向训练好的 checkpoint；result_dir 不传时使用配置文件里的默认输出目录。
    parser.add_argument("--weights", required=True)
    parser.add_argument("--result_dir")
    # 如果提供 gt_dir，就会按文件名配对 GT，并额外计算 PSNR/SSIM。
    parser.add_argument("--gt_dir")
    # 命令行开关，用于临时关闭配置文件中的 self-ensemble/TTA 推理。
    parser.add_argument("--no_self_ensemble", action="store_true")
    return parser.parse_args()


def build_model(config, device):
    """根据 YAML 配置创建 SG_LLIE 模型，并移动到指定设备。"""
    # 配置里的 name 只是模型名称标识，不是 SG_LLIE 构造函数参数。
    model_config = dict(config["model"])
    model_config.pop("name", None)
    return SG_LLIE(**model_config).to(device)


def main():
    args = parse_args()
    # 先检查必需路径参数，避免后面读文件时才报不清楚的错误。
    require_keys(vars(args), ["input_dir", "input_s_dir", "weights"], "testing paths")

    config = ConfigLoader(args.config).load()
    result_dir = ensure_dir(args.result_dir or config["paths"].get("result_dir", ROOT / "test" / "results"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 根据配置创建模型、加载权重、切换到 eval 模式。
    model = build_model(config, device)
    load_checkpoint(args.weights, model, device=device, strict=False)
    model.eval()

    # 按文件名把低照度输入和结构先验配对；如果有 GT，也按同名文件建立查找表。
    pairs = paired_by_name(args.input_dir, args.input_s_dir)
    gt_pairs = dict((Path(src).name, gt) for src, gt in paired_by_name(args.input_dir, args.gt_dir)) if args.gt_dir else {}

    # self-ensemble 是测试时增强(TTA)：翻转/旋转后多次推理再平均。
    # 默认关闭，只有配置文件显式写 true 且命令行没有 --no_self_ensemble 时才启用。
    use_ensemble = config["testing"].get("self_ensemble", False) and not args.no_self_ensemble
    # 为了兼容网络下采样/上采样，推理前把图像 padding 到 factor 的倍数。
    factor = config["testing"].get("factor", 32)

    metric_sums = {"psnr": 0.0, "ssim": 0.0}
    metric_count = 0
    with torch.no_grad():
        for input_path, prior_path in tqdm(pairs, desc="test"):
            # 读取图像并转为 [1, 3, H, W]，数值范围为 [0, 1]。
            image = image_to_tensor(load_image(input_path)).unsqueeze(0).to(device)
            prior = image_to_tensor(load_image(prior_path)).unsqueeze(0).to(device)

            # 记录原始尺寸，padding 后推理，最后再裁回原图大小。
            image, (h, w) = pad_to_factor(image, factor)
            prior, _ = pad_to_factor(prior, factor)

            restored = self_ensemble(image, prior, model) if use_ensemble else model(image, prior)[0]
            restored = restored[:, :, :h, :w].clamp(0.0, 1.0)

            # 保存增强结果，文件名沿用输入图像文件名。
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
