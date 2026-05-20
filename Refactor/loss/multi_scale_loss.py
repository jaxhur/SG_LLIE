"""Combined multi-scale loss for SG_LLIE outputs."""

import torch.nn as nn
import torch.nn.functional as F

from loss.charbonnier import CharbonnierLoss
from loss.msssim import ms_ssim
from loss.perceptual import VGGPerceptualLoss


class SGLLIEMultiScaleLoss(nn.Module):
    """Combine Charbonnier, optional VGG perceptual, and optional MS-SSIM losses."""

    def __init__(
        self,
        charbonnier_weight=1.0,
        perceptual_weight=0.01,
        msssim_weight=0.4,
        use_pretrained_vgg=False,
        feature_layers=None,
    ):
        """Store weights and create sub-loss modules for all SG_LLIE output scales."""
        super().__init__()
        self.charbonnier_weight = charbonnier_weight
        self.perceptual_weight = perceptual_weight
        self.msssim_weight = msssim_weight
        self.charbonnier = CharbonnierLoss()
        self.perceptual = (
            VGGPerceptualLoss(use_pretrained=use_pretrained_vgg, feature_layers=feature_layers)
            if perceptual_weight > 0
            else None
        )

    def forward(self, outputs, target):
        """Return total scalar loss and a log dictionary for three SG_LLIE predictions."""
        out1, out2, out3 = outputs
        targets = [
            target,
            F.interpolate(target, scale_factor=0.5, mode="bilinear", align_corners=False),
            F.interpolate(target, scale_factor=0.25, mode="bilinear", align_corners=False),
        ]
        predictions = [out1, out2, out3]
        charbonnier_loss = sum(self.charbonnier(pred, gt) for pred, gt in zip(predictions, targets))
        perceptual_loss = target.new_tensor(0.0)
        if self.perceptual is not None:
            perceptual_loss = sum(self.perceptual(pred, gt) for pred, gt in zip(predictions, targets))
        msssim_loss = target.new_tensor(0.0)
        if self.msssim_weight > 0:
            msssim_loss = sum(1.0 - ms_ssim(pred, gt, normalize=True) for pred, gt in zip(predictions, targets))
        total = (
            self.charbonnier_weight * charbonnier_loss
            + self.perceptual_weight * perceptual_loss
            + self.msssim_weight * msssim_loss
        )
        log_dict = {
            "loss": total.detach(),
            "charbonnier": charbonnier_loss.detach(),
            "perceptual": perceptual_loss.detach(),
            "msssim": msssim_loss.detach(),
        }
        return total, log_dict
