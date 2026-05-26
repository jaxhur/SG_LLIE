"""Charbonnier 重建损失。"""

import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):
    """图像复原中常用的平滑 L1 损失，比普通 L1 更稳定。"""

    def __init__(self, eps=1e-3):
        """初始化损失函数，eps 用于避免 sqrt(0) 的数值问题。"""
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        """计算 pred 和 target 之间的平均 Charbonnier 损失。"""
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
