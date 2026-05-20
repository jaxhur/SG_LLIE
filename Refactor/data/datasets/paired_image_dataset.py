"""Paired image dataset for SG_LLIE training and validation."""

from torch.utils.data import Dataset

from data.transforms import augment_geometric, paired_random_crop, reflect_pad_to_size
from utils.image_io import image_to_tensor, load_image
from utils.paths import paired_by_name


class PairedImageDataset(Dataset):
    """Load paired low-light, ground-truth, and structure-prior image folders."""

    def __init__(
        self,
        lq_dir,
        gt_dir,
        lq_s_dir,
        gt_s_dir=None,
        phase="train",
        crop_size=None,
        geometric_augs=False,
    ):
        """Create folder pairs and store augmentation options for the requested phase."""
        self.phase = phase
        self.crop_size = crop_size
        self.geometric_augs = geometric_augs
        self.lq_gt_pairs = paired_by_name(lq_dir, gt_dir)
        if gt_s_dir is None:
            self.lq_s_pairs = [(path, None) for path, _ in paired_by_name(lq_dir, lq_s_dir)]
        else:
            self.lq_s_pairs = paired_by_name(lq_s_dir, gt_s_dir)
        if len(self.lq_gt_pairs) != len(self.lq_s_pairs):
            raise ValueError("Image pairs and structure-prior pairs have different lengths.")

    def __len__(self):
        """Return the number of paired samples."""
        return len(self.lq_gt_pairs)

    def __getitem__(self, index):
        """Load one sample and return tensors plus source paths."""
        lq_path, gt_path = self.lq_gt_pairs[index]
        lq_s_path, gt_s_path = self.lq_s_pairs[index]
        lq = load_image(lq_path)
        gt = load_image(gt_path)
        lq_s = load_image(lq_s_path)
        gt_s = load_image(gt_s_path) if gt_s_path is not None else lq_s.copy()
        images = [lq, gt, lq_s, gt_s]
        if self.phase == "train" and self.crop_size:
            images = reflect_pad_to_size(images, self.crop_size)
            images = paired_random_crop(images, self.crop_size)
            images = augment_geometric(images, self.geometric_augs)
        lq, gt, lq_s, gt_s = images
        return {
            "lq": image_to_tensor(lq),
            "gt": image_to_tensor(gt),
            "lq_s": image_to_tensor(lq_s),
            "gt_s": image_to_tensor(gt_s),
            "lq_path": str(lq_path),
            "gt_path": str(gt_path),
            "lq_s_path": str(lq_s_path),
        }
