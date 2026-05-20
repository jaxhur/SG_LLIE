"""Standalone SG_LLIE training entrypoint."""

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
    """Parse CLI arguments whose paths should override YAML configuration."""
    parser = argparse.ArgumentParser(description="Train SG_LLIE without external restoration frameworks.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "sg_llie_ntire25.yaml"))
    parser.add_argument("--train_lq_dir", required=True)
    parser.add_argument("--train_gt_dir", required=True)
    parser.add_argument("--train_lq_s_dir", required=True)
    parser.add_argument("--train_gt_s_dir", required=True)
    parser.add_argument("--val_lq_dir")
    parser.add_argument("--val_gt_dir")
    parser.add_argument("--val_lq_s_dir")
    parser.add_argument("--val_gt_s_dir")
    parser.add_argument("--checkpoint_dir")
    parser.add_argument("--resume")
    parser.add_argument("--log_dir")
    return parser.parse_args()


def build_model(config, device):
    """Instantiate SG_LLIE from config and move it to `device`."""
    model_config = dict(config["model"])
    model_config.pop("name", None)
    model = SG_LLIE(**model_config)
    return model.to(device)


def build_optimizer(config, model):
    """Create the configured optimizer for trainable model parameters."""
    optim_config = dict(config["optimizer"])
    optim_type = optim_config.pop("type", "Adam")
    if optim_type == "Adam":
        return torch.optim.Adam(model.parameters(), **optim_config)
    if optim_type == "AdamW":
        return torch.optim.AdamW(model.parameters(), **optim_config)
    raise ValueError(f"Unsupported optimizer type: {optim_type}")


def main():
    """Run the full SG_LLIE training loop with optional validation."""
    args = parse_args()
    config = ConfigLoader(args.config).load()
    set_random_seed(config["training"].get("seed", 100))
    cli_paths = vars(args)
    require_keys(cli_paths, ["train_lq_dir", "train_gt_dir", "train_lq_s_dir", "train_gt_s_dir"], "training paths")
    checkpoint_dir = ensure_dir(args.checkpoint_dir or config["paths"].get("checkpoint_dir", ROOT / "checkpoints"))
    log_dir = ensure_dir(args.log_dir or config["logging"].get("log_dir", ROOT / "logger" / "logs"))
    logger = create_logger("SG_LLIE", log_dir=log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    train_loader = build_train_dataloader(config, cli_paths)
    val_loader = build_val_dataloader(config, cli_paths)
    model = build_model(config, device)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(optimizer, config["scheduler"])
    criterion = SGLLIEMultiScaleLoss(**config["loss"]).to(device)
    start_iter = 0
    best_psnr = None
    if args.resume:
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
    data_iter = itertools.cycle(train_loader)
    for iteration in range(start_iter + 1, total_iter + 1):
        batch = next(data_iter)
        lq = batch["lq"].to(device)
        gt = batch["gt"].to(device)
        lq_s = batch["lq_s"].to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(lq, lq_s)
        loss, log_dict = criterion(outputs, gt)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        if iteration % print_freq == 0:
            lr = optimizer.param_groups[0]["lr"]
            message_logger.log(iteration, lr, log_dict)
            if tb_writer is not None:
                tb_writer.add_scalar("train/lr", lr, iteration)
                for key, value in log_dict.items():
                    tb_writer.add_scalar(f"train/{key}", float(value), iteration)
        if iteration % save_freq == 0:
            save_checkpoint(checkpoint_dir / f"iter_{iteration}.pth", model, optimizer, scheduler, iteration, best_psnr)
            save_checkpoint(checkpoint_dir / "latest.pth", model, optimizer, scheduler, iteration, best_psnr)
            logger.info("Saved checkpoint at iteration %d", iteration)
        if val_loader is not None and iteration % val_freq == 0:
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
                best_psnr = metrics["psnr"]
                save_checkpoint(checkpoint_dir / "best_psnr.pth", model, optimizer, scheduler, iteration, best_psnr)
                logger.info("Saved new best PSNR checkpoint: %.4f", best_psnr)
    save_checkpoint(checkpoint_dir / "latest.pth", model, optimizer, scheduler, total_iter, best_psnr)
    if tb_writer is not None:
        tb_writer.close()


if __name__ == "__main__":
    main()
