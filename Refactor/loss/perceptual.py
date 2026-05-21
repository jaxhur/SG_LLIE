"""VGG-based perceptual loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGPerceptualLoss(nn.Module):
    """Compare images using selected VGG16 feature layers."""

    def __init__(self, resize=True, use_pretrained=False, feature_layers=None):
        """Create frozen VGG16 feature blocks and ImageNet normalization tensors."""
        super().__init__()
        try:
            from torchvision import models
        except ImportError as exc:
            raise ImportError(
                "VGG perceptual loss requires torchvision and a working Pillow installation. "
                "Either reinstall Pillow/torchvision or set loss.perceptual_weight to 0.0."
            ) from exc
        self.feature_layers = feature_layers if feature_layers is not None else [2]
        weights = models.VGG16_Weights.IMAGENET1K_V1 if use_pretrained else None
        features = models.vgg16(weights=weights).features
        self.blocks = nn.ModuleList(
            [
                features[:4].eval(),
                features[4:9].eval(),
                features[9:16].eval(),
                features[16:23].eval(),
            ]
        )
        for block in self.blocks:
            for parameter in block.parameters():
                parameter.requires_grad = False
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        """Return L1 distance between selected VGG features of `pred` and `target`."""
        if pred.shape[1] != 3:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            pred = F.interpolate(pred, size=(224, 224), mode="bilinear", align_corners=False)
            target = F.interpolate(target, size=(224, 224), mode="bilinear", align_corners=False)
        loss = pred.new_tensor(0.0)
        x = pred
        y = target
        for index, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if index in self.feature_layers:
                loss = loss + F.l1_loss(x, y)
        return loss
