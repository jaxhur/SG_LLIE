"""Charbonnier reconstruction loss."""

import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):
    """Smooth L1-like loss commonly used for image restoration."""

    def __init__(self, eps=1e-3):
        """Store numerical stability constant `eps`."""
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        """Return mean Charbonnier loss between predicted and target BCHW tensors."""
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
