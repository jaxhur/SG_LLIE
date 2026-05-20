"""Learning-rate schedulers used by the standalone training loop."""

import math

from torch.optim.lr_scheduler import _LRScheduler


def get_position_from_periods(iteration, cumulative_period):
    """Return the scheduler cycle index for `iteration` given cumulative periods."""
    for index, period in enumerate(cumulative_period):
        if iteration <= period:
            return index
    return len(cumulative_period) - 1


class CosineAnnealingRestartCyclicLR(_LRScheduler):
    """Cosine annealing scheduler with cycle-specific minimum learning rates."""

    def __init__(self, optimizer, periods, restart_weights=(1,), eta_mins=(0,), last_epoch=-1):
        """Store cycle periods, restart weights, eta minimums, and initialize PyTorch scheduler state."""
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
        """Compute the current learning rate for each optimizer parameter group."""
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
    """Create a scheduler from a YAML config dictionary and return it."""
    scheduler_type = scheduler_config.get("type", "CosineAnnealingRestartCyclicLR")
    if scheduler_type != "CosineAnnealingRestartCyclicLR":
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    return CosineAnnealingRestartCyclicLR(
        optimizer,
        periods=scheduler_config["periods"],
        restart_weights=scheduler_config.get("restart_weights", [1] * len(scheduler_config["periods"])),
        eta_mins=scheduler_config.get("eta_mins", [0] * len(scheduler_config["periods"])),
    )
