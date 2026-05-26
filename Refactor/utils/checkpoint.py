"""模型 checkpoint 保存和加载工具。"""

from pathlib import Path

import torch

from utils.paths import ensure_dir


def save_checkpoint(path, model, optimizer=None, scheduler=None, iteration=0, best_metric=None):
    """保存训练状态。

    输入:
        path: 保存路径。
        model: SG_LLIE 模型。
        optimizer: 可选优化器状态。
        scheduler: 可选学习率调度器状态。
        iteration: 当前迭代数。
        best_metric: 当前最好指标，例如 best_psnr。
    输出:
        无返回值，写出 .pth 文件。
    """
    path = Path(path)
    ensure_dir(path.parent)
    payload = {
        "model_name": "SG_LLIE",
        "iteration": iteration,
        "params": model.state_dict(),
        "best_metric": best_metric,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu", strict=True):
    """加载训练状态。

    输入:
        path: checkpoint 路径。
        model: 需要加载参数的模型。
        optimizer/scheduler: 如果提供，则一起恢复训练状态。
        device: map_location 设备。
        strict: 是否严格匹配模型参数名。
    输出:
        checkpoint 原始字典。
    """
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("params", checkpoint)
    model.load_state_dict(state_dict, strict=strict)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint
