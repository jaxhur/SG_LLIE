"""SG_LLIE 训练和验证使用的成对图像数据集。

每个样本包含:
    lq: 低照度输入图像。
    gt: 正常曝光 GT 图像。
    lq_s: 低照度输入对应的结构先验。
    gt_s: 可选字段，当前训练逻辑不使用，仅为了兼容可能的扩展。
"""

from torch.utils.data import Dataset

from data.transforms import augment_geometric, paired_random_crop, reflect_pad_to_size
from utils.image_io import image_to_tensor, load_image
from utils.paths import paired_by_name


class PairedImageDataset(Dataset):
    """从文件夹读取低照度图像、GT 图像和结构先验，并保证它们按文件名配对。"""

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
        """初始化数据集。

        输入参数:
            lq_dir: 低照度输入图像目录。
            gt_dir: GT 图像目录。
            lq_s_dir: 低照度输入对应的结构先验目录。
            gt_s_dir: GT 结构先验目录，可选，当前训练不依赖。
            phase: "train" 时会启用训练增强，"val"/"test" 时不做随机增强。
            crop_size: 随机裁剪大小；为 None 时使用整图训练。
            geometric_augs: 是否启用随机翻转、旋转等几何增强。
        输出:
            无返回值，内部建立文件路径配对列表。
        """
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
        """返回数据集中样本数量。"""
        return len(self.lq_gt_pairs)

    def __getitem__(self, index):
        """读取一个样本。

        输入:
            index: 样本索引。
        输出:
            字典，包含 lq/gt/lq_s/gt_s 张量，以及对应文件路径。
        作用:
            训练阶段会按配置执行随机裁剪和几何增强；验证/测试阶段保持原图。
        """
        lq_path, gt_path = self.lq_gt_pairs[index]
        lq_s_path, gt_s_path = self.lq_s_pairs[index]
        lq = load_image(lq_path)
        gt = load_image(gt_path)
        lq_s = load_image(lq_s_path)
        gt_s = load_image(gt_s_path) if gt_s_path is not None else lq_s.copy()
        images = [lq, gt, lq_s, gt_s]
        if self.phase == "train":
            if self.crop_size:
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
