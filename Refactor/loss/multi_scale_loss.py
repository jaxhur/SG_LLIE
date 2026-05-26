"""SG_LLIE 的多尺度组合损失。

模型会输出三个尺度的结果，这里分别和 GT、1/2 GT、1/4 GT 计算损失，
再按权重组合成最终反向传播使用的标量 loss。
"""

import torch.nn as nn
import torch.nn.functional as F

from loss.charbonnier import CharbonnierLoss
from loss.msssim import ms_ssim
from loss.perceptual import VGGPerceptualLoss


class SGLLIEMultiScaleLoss(nn.Module):
    """组合 Charbonnier、VGG 感知损失和 MS-SSIM 损失。"""

    def __init__(
        self,
        charbonnier_weight=1.0,
        perceptual_weight=0.01,
        msssim_weight=0.4,
        use_pretrained_vgg=False,
        feature_layers=None,
    ):
        """初始化多尺度损失。

        输入参数:
            charbonnier_weight: Charbonnier 损失权重。
            perceptual_weight: VGG 感知损失权重，为 0 时不启用 torchvision。
            msssim_weight: MS-SSIM 损失权重。
            use_pretrained_vgg: 是否使用 ImageNet 预训练 VGG。
            feature_layers: 使用哪些 VGG 层作为感知特征。
        输出:
            无返回值。
        """
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
        """计算总损失。

        输入:
            outputs: 模型三个尺度输出 (out1, out2, out3)。
            target: 原尺度 GT 图像 [B, 3, H, W]。
        输出:
            total: 用于 backward 的标量损失。
            log_dict: 用于日志打印的各项损失。
        """
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
