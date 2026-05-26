"""SG_LLIE 独立训练入口。

这个脚本不依赖外部训练框架，负责:
1. 读取 YAML 配置；
2. 读取命令行传入的数据路径；
3. 构建模型、损失函数、优化器、学习率策略；
4. 周期性打印日志、保存 checkpoint、执行验证。
"""

import argparse
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from data.dataloader import build_train_dataloader, build_val_dataloader
from logger.logger import MessageLogger, create_logger, create_tb_writer
from loss import SGLLIEMultiScaleLoss
from model import SG_LLIE
from train.validate import validate
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.config import ConfigLoader, require_keys
from utils.paths import ensure_dir
from utils.scheduler import build_scheduler
from utils.seed import set_random_seed


def parse_args():
    """解析命令行参数
    """
    parser = argparse.ArgumentParser(description="Train SG_LLIE without external restoration frameworks.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "sg_llie_ntire25.yaml"))
    parser.add_argument("--train_lq_dir", required=True)
    parser.add_argument("--train_gt_dir", required=True)
    parser.add_argument("--train_lq_s_dir", required=True)
    parser.add_argument("--train_gt_s_dir")
    parser.add_argument("--val_lq_dir")
    parser.add_argument("--val_gt_dir")
    parser.add_argument("--val_lq_s_dir")
    parser.add_argument("--val_gt_s_dir")
    parser.add_argument("--checkpoint_dir")
    parser.add_argument("--resume")
    parser.add_argument("--log_dir")
    return parser.parse_args()


def build_model(config, device):
    """根据 YAML 配置创建 SG_LLIE 模型，并移动到指定设备。"""
    model_config = dict(config["model"])
    model_config.pop("name", None)
    model = SG_LLIE(**model_config)
    return model.to(device)


def build_optimizer(config, model):
    """根据 YAML 配置创建优化器。
    输入:
        config: 配置字典。
        model: 需要训练的 SG_LLIE 模型。
    输出:
        torch.optim 优化器实例。
    """
    optim_config = dict(config["optimizer"])
    optim_type = optim_config.pop("type", "Adam")
    if optim_type == "Adam":
        return torch.optim.Adam(model.parameters(), **optim_config)
    if optim_type == "AdamW":
        return torch.optim.AdamW(model.parameters(), **optim_config)
    raise ValueError(f"Unsupported optimizer type: {optim_type}")


def main():
    """执行完整训练流程
    输入:
        所有参数来自命令行和 YAML。
    输出:
        训练过程中会写日志、保存 checkpoint
    """
    args = parse_args()
    config = ConfigLoader(args.config).load()
    set_random_seed(config["training"].get("seed", 100))

    # 命令行只负责数据/输出路径；训练超参数从 YAML 读取。
    cli_paths = vars(args)
    require_keys(cli_paths, ["train_lq_dir", "train_gt_dir", "train_lq_s_dir"], "training paths")
    checkpoint_dir = ensure_dir(args.checkpoint_dir or config["paths"].get("checkpoint_dir", ROOT / "checkpoints"))
    log_dir = ensure_dir(args.log_dir or config["logging"].get("log_dir", ROOT / "logger" / "logs"))

    logger = create_logger("SG_LLIE", log_dir=log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # 构建训练/验证数据、模型、优化器、调度器和多尺度损失。
    train_loader = build_train_dataloader(config, cli_paths)
    val_loader = build_val_dataloader(config, cli_paths)
    model = build_model(config, device)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(optimizer, config["scheduler"])
    criterion = SGLLIEMultiScaleLoss(**config["loss"]).to(device)

    start_iter = 0
    best_psnr = None
    if args.resume:
        # 断点续训时同时恢复模型、优化器、学习率调度器和最好指标。
        checkpoint = load_checkpoint(args.resume, model, optimizer=optimizer, scheduler=scheduler, device=device, strict=False)
        start_iter = int(checkpoint.get("iteration", 0))
        best_psnr = checkpoint.get("best_metric")
        logger.info("Resumed from %s at iteration %d", args.resume, start_iter)

    total_iter = config["training"]["total_iter"]
    print_freq = config["logging"].get("print_freq", 20)
    save_freq = config["logging"].get("save_checkpoint_freq", 1000)
    val_freq = config["validation"].get("val_freq", 5000)
    grad_clip = config["training"].get("grad_clip")
    message_logger = MessageLogger(logger, total_iter)
    tb_writer = create_tb_writer(log_dir, config["logging"].get("use_tb_logger", False))

    model.train()
    # cycle 让 dataloader 用完后自动从头开始，直到达到 total_iter。
    data_iter = itertools.cycle(train_loader)
    for iteration in range(start_iter + 1, total_iter + 1):
        batch = next(data_iter)
        lq = batch["lq"].to(device)
        gt = batch["gt"].to(device)
        lq_s = batch["lq_s"].to(device)

        # 标准训练步骤：前向、多尺度损失、反传、可选梯度裁剪、更新参数。
        optimizer.zero_grad(set_to_none=True)
        outputs = model(lq, lq_s)
        loss, log_dict = criterion(outputs, gt)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        if iteration % print_freq == 0:
            # 训练日志同时写普通 logger 和 TensorBoard。
            lr = optimizer.param_groups[0]["lr"]
            message_logger.log(iteration, lr, log_dict)
            if tb_writer is not None:
                tb_writer.add_scalar("train/lr", lr, iteration)
                for key, value in log_dict.items():
                    tb_writer.add_scalar(f"train/{key}", float(value), iteration)

        if iteration % save_freq == 0:
            # 定期保存当前迭代 checkpoint，并覆盖 latest.pth 方便续训。
            save_checkpoint(checkpoint_dir / f"iter_{iteration}.pth", model, optimizer, scheduler, iteration, best_psnr)
            save_checkpoint(checkpoint_dir / "latest.pth", model, optimizer, scheduler, iteration, best_psnr)
            logger.info("Saved checkpoint at iteration %d", iteration)

        if val_loader is not None and iteration % val_freq == 0:
            # 验证阶段返回平均 PSNR/SSIM，并可保存验证可视化结果。
            metrics = validate(
                model,
                val_loader,
                device,
                iteration,
                save_dir=checkpoint_dir / "validation_images",
                save_images=config["validation"].get("save_img", True),
            )
            logger.info("Validation iter %d PSNR %.4f SSIM %.4f", iteration, metrics["psnr"], metrics["ssim"])
            if tb_writer is not None:
                for key, value in metrics.items():
                    tb_writer.add_scalar(f"val/{key}", value, iteration)
            if best_psnr is None or metrics["psnr"] > best_psnr:
                # 只按 PSNR 选择 best checkpoint。
                best_psnr = metrics["psnr"]
                save_checkpoint(checkpoint_dir / "best_psnr.pth", model, optimizer, scheduler, iteration, best_psnr)
                logger.info("Saved new best PSNR checkpoint: %.4f", best_psnr)

    # 训练结束后再保存一次 latest，保证最终状态落盘。
    save_checkpoint(checkpoint_dir / "latest.pth", model, optimizer, scheduler, total_iter, best_psnr)
    if tb_writer is not None:
        tb_writer.close()

if __name__ == "__main__":
    main()
