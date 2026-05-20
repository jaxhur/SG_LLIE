"""Console, file, and optional TensorBoard logging helpers."""

import logging
from pathlib import Path

from utils.paths import ensure_dir


def create_logger(name, log_dir=None, filename="train.log"):
    """Create a logger that writes to stdout and optionally to `log_dir/filename`."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if log_dir is not None:
        log_dir = ensure_dir(log_dir)
        file_handler = logging.FileHandler(Path(log_dir) / filename, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


class MessageLogger:
    """Format periodic training loss and learning-rate messages."""

    def __init__(self, logger, total_iter):
        """Store a Python logger and the configured total iteration count."""
        self.logger = logger
        self.total_iter = total_iter

    def log(self, iteration, lr, loss_dict):
        """Write a single training progress line with scalar losses from `loss_dict`."""
        losses = " ".join(f"{key}: {float(value):.6f}" for key, value in loss_dict.items())
        self.logger.info("iter: %d/%d lr: %.8f %s", iteration, self.total_iter, lr, losses)


def create_tb_writer(log_dir, enabled):
    """Return a TensorBoard SummaryWriter when enabled, otherwise return `None`."""
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise ImportError("TensorBoard logging requested but tensorboard is not installed.") from exc
    return SummaryWriter(log_dir=str(log_dir))
