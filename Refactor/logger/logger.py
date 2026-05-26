"""日志工具，支持控制台、文件和可选 TensorBoard。"""

import logging
from pathlib import Path

from utils.paths import ensure_dir


def create_logger(name, log_dir=None, filename="train.log"):
    """创建 logger。

    输入:
        name: logger 名称。
        log_dir: 日志保存目录，可为空。
        filename: 日志文件名。
    输出:
        logging.Logger 实例。
    """
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
    """格式化训练过程中的 loss 和学习率日志。"""

    def __init__(self, logger, total_iter):
        """保存 logger 和总迭代数。"""
        self.logger = logger
        self.total_iter = total_iter

    def log(self, iteration, lr, loss_dict):
        """打印一条训练日志，包含 iteration、lr 和各项 loss。"""
        losses = " ".join(f"{key}: {float(value):.6f}" for key, value in loss_dict.items())
        self.logger.info("iter: %d/%d lr: %.8f %s", iteration, self.total_iter, lr, losses)


def create_tb_writer(log_dir, enabled):
    """根据配置创建 TensorBoard writer；未启用时返回 None。"""
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise ImportError("TensorBoard logging requested but tensorboard is not installed.") from exc
    return SummaryWriter(log_dir=str(log_dir))
