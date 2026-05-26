"""DataLoader 构建函数。

训练脚本只负责解析路径参数，这里负责把路径和 YAML 配置组合成 PyTorch DataLoader。
"""

from torch.utils.data import DataLoader

from data.datasets import PairedImageDataset


def build_train_dataloader(config, paths):
    """构建训练 DataLoader。

    输入:
        config: YAML 解析出的配置字典。
        paths: argparse 得到的路径字典，必须包含 train_lq_dir/train_gt_dir/train_lq_s_dir。
    输出:
        PyTorch DataLoader，每个 batch 包含 lq、gt、lq_s 等张量。
    """
    dataset = PairedImageDataset(
        lq_dir=paths["train_lq_dir"],
        gt_dir=paths["train_gt_dir"],
        lq_s_dir=paths["train_lq_s_dir"],
        gt_s_dir=paths.get("train_gt_s_dir"),
        phase="train",
        crop_size=config["training"].get("gt_size"),
        geometric_augs=config.get("augmentation", {}).get("geometric", True),
    )
    return DataLoader(
        dataset,
        batch_size=config["training"].get("batch_size", 2),
        shuffle=True,
        num_workers=config["training"].get("num_workers", 2),
        pin_memory=True,
        drop_last=True,
    )


def build_val_dataloader(config, paths):
    """构建验证 DataLoader。

    输入:
        config: YAML 配置。
        paths: 路径字典；只有 val_lq_dir/val_gt_dir/val_lq_s_dir 都存在时才启用验证。
    输出:
        验证 DataLoader；如果验证路径不完整，则返回 None。
    """
    required = ["val_lq_dir", "val_gt_dir", "val_lq_s_dir"]
    if not all(paths.get(key) for key in required):
        return None
    dataset = PairedImageDataset(
        lq_dir=paths["val_lq_dir"],
        gt_dir=paths["val_gt_dir"],
        lq_s_dir=paths["val_lq_s_dir"],
        gt_s_dir=paths.get("val_gt_s_dir"),
        phase="val",
    )
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config["validation"].get("num_workers", 1),
        pin_memory=True,
        drop_last=False,
    )
