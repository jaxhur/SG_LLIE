"""训练过程中使用的验证函数。"""

from pathlib import Path

import torch
from tqdm import tqdm

from metrics.psnr_ssim import calculate_psnr, calculate_ssim
from utils.image_io import save_image, tensor_to_image
from utils.paths import ensure_dir


def validate(model, dataloader, device, iteration, save_dir=None, save_images=True):
    """执行一次验证。

    输入:
        model: 当前训练中的 SG_LLIE 模型。
        dataloader: 验证集 DataLoader。
        device: cuda 或 cpu。
        iteration: 当前训练迭代数，用于日志和保存目录命名。
        save_dir: 验证结果图片保存目录。
        save_images: 是否保存验证输出图。
    输出:
        字典，包含平均 PSNR 和 SSIM。
    """
    model.eval()
    metric_sums = {"psnr": 0.0, "ssim": 0.0}
    count = 0
    if save_images and save_dir is not None:
        save_dir = ensure_dir(Path(save_dir) / str(iteration))
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="validate", leave=False):
            lq = batch["lq"].to(device)
            gt = batch["gt"].to(device)
            lq_s = batch["lq_s"].to(device)
            pred = model(lq, lq_s)[0].clamp(0.0, 1.0)
            metric_sums["psnr"] += calculate_psnr(pred, gt)
            metric_sums["ssim"] += calculate_ssim(pred, gt)
            if save_images and save_dir is not None:
                image_name = Path(batch["lq_path"][0]).name
                save_image(save_dir / image_name, tensor_to_image(pred))
            count += 1
    model.train()
    if count == 0:
        return {"psnr": 0.0, "ssim": 0.0}
    return {key: value / count for key, value in metric_sums.items()}
