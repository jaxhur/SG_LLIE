"""DataLoader construction helpers."""

from torch.utils.data import DataLoader

from data.datasets import PairedImageDataset


def build_train_dataloader(config, paths):
    """Build and return the training DataLoader from YAML config and CLI paths."""
    dataset = PairedImageDataset(
        lq_dir=paths["train_lq_dir"],
        gt_dir=paths["train_gt_dir"],
        lq_s_dir=paths["train_lq_s_dir"],
        gt_s_dir=paths["train_gt_s_dir"],
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
    """Build a validation DataLoader when validation paths exist; otherwise return `None`."""
    required = ["val_lq_dir", "val_gt_dir", "val_lq_s_dir", "val_gt_s_dir"]
    if not all(paths.get(key) for key in required):
        return None
    dataset = PairedImageDataset(
        lq_dir=paths["val_lq_dir"],
        gt_dir=paths["val_gt_dir"],
        lq_s_dir=paths["val_lq_s_dir"],
        gt_s_dir=paths["val_gt_s_dir"],
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
