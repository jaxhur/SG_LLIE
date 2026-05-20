"""Checkpoint save and load helpers."""

from pathlib import Path

import torch

from utils.paths import ensure_dir


def save_checkpoint(path, model, optimizer=None, scheduler=None, iteration=0, best_metric=None):
    """Save model state and optional optimizer/scheduler states to `path`."""
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
    """Load model state from `path` and optionally restore optimizer and scheduler."""
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("params", checkpoint)
    model.load_state_dict(state_dict, strict=strict)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint
