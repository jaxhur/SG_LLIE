"""可选 LPIPS 指标封装。"""

import torch


class LPIPSMetric:
    """在安装 lpips 包时计算 LPIPS 感知距离。"""

    def __init__(self, net="alex", device="cuda"):
        """初始化 LPIPS 模型。
        输入参数:
            net: LPIPS 使用的主干网络名称，例如 alex。
        输出:
            无返回值。如果未安装 lpips，会给出清晰错误。
        """
        try:
            import lpips
        except ImportError as exc:
            raise ImportError("LPIPS metric requested but package `lpips` is not installed.") from exc
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = lpips.LPIPS(net=net).to(self.device)
        self.model.eval()

    def __call__(self, pred, target):
        """输入 pred 和 target，返回 LPIPS 距离；数值越小表示感知上越接近。"""
        pred = pred.to(self.device) * 2.0 - 1.0
        target = target.to(self.device) * 2.0 - 1.0
        with torch.no_grad():
            return float(self.model(pred, target).mean().item())
