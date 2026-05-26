"""学习率调度器。"""

import math

from torch.optim.lr_scheduler import _LRScheduler


def get_position_from_periods(iteration, cumulative_period):
    """根据当前迭代数找到它属于第几个学习率周期。"""
    for index, period in enumerate(cumulative_period):
        if iteration <= period:
            return index
    return len(cumulative_period) - 1


class CosineAnnealingRestartCyclicLR(_LRScheduler):
    """带重启的余弦退火学习率策略，每个周期可以有不同最小学习率。"""

    def __init__(self, optimizer, periods, restart_weights=(1,), eta_mins=(0,), last_epoch=-1):
        """初始化调度器。

        输入参数:
            optimizer: PyTorch 优化器。
            periods: 每个余弦周期的迭代数。
            restart_weights: 每次重启后的学习率权重。
            eta_mins: 每个周期的最小学习率。
            last_epoch: PyTorch 调度器内部状态。
        """
        if len(periods) != len(restart_weights):
            raise ValueError("periods and restart_weights must have the same length.")
        if len(periods) != len(eta_mins):
            raise ValueError("periods and eta_mins must have the same length.")
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_mins = eta_mins
        self.cumulative_period = [sum(periods[: i + 1]) for i in range(len(periods))]
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """根据当前迭代数计算每个参数组的学习率。"""
        idx = get_position_from_periods(self.last_epoch, self.cumulative_period)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        current_period = self.periods[idx]
        eta_min = self.eta_mins[idx]
        return [
            eta_min
            + current_weight
            * 0.5
            * (base_lr - eta_min)
            * (1 + math.cos(math.pi * ((self.last_epoch - nearest_restart) / current_period)))
            for base_lr in self.base_lrs
        ]


def build_scheduler(optimizer, scheduler_config):
    """根据 YAML 配置创建学习率调度器。"""
    scheduler_type = scheduler_config.get("type", "CosineAnnealingRestartCyclicLR")
    if scheduler_type != "CosineAnnealingRestartCyclicLR":
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    return CosineAnnealingRestartCyclicLR(
        optimizer,
        periods=scheduler_config["periods"],
        restart_weights=scheduler_config.get("restart_weights", [1] * len(scheduler_config["periods"])),
        eta_mins=scheduler_config.get("eta_mins", [0] * len(scheduler_config["periods"])),
    )
