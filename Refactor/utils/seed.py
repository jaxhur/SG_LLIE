"""随机种子工具，用于提高实验可复现性。"""

import random

import numpy as np
import torch


def set_random_seed(seed):
    """设置 Python、NumPy、PyTorch 的随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
