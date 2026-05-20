"""Optional LPIPS metric wrapper."""

import torch


class LPIPSMetric:
    """Compute LPIPS distance when the optional `lpips` package is installed."""

    def __init__(self, net="alex", device="cuda"):
        """Create an LPIPS model on `device`; raise a clear error if dependency is missing."""
        try:
            import lpips
        except ImportError as exc:
            raise ImportError("LPIPS metric requested but package `lpips` is not installed.") from exc
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = lpips.LPIPS(net=net).to(self.device)
        self.model.eval()

    def __call__(self, pred, target):
        """Return LPIPS distance for BCHW tensors in `[0, 1]`."""
        pred = pred.to(self.device) * 2.0 - 1.0
        target = target.to(self.device) * 2.0 - 1.0
        with torch.no_grad():
            return float(self.model(pred, target).mean().item())
