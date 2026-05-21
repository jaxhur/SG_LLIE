import importlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class RefactorSmokeTest(unittest.TestCase):
    def test_model_forward_returns_three_scales(self):
        module = importlib.import_module("model.sg_llie")
        model = module.SG_LLIE(en_feature_num=8, en_inter_num=4, de_feature_num=8, de_inter_num=4, sam_number=1)
        x = torch.rand(1, 3, 64, 64)
        s = torch.rand(1, 3, 64, 64)

        out1, out2, out3 = model(x, s)

        self.assertEqual(tuple(out1.shape), (1, 3, 64, 64))
        self.assertEqual(tuple(out2.shape), (1, 3, 32, 32))
        self.assertEqual(tuple(out3.shape), (1, 3, 16, 16))

    def test_multiscale_loss_returns_scalar_and_log_dict(self):
        module = importlib.import_module("loss.multi_scale_loss")
        criterion = module.SGLLIEMultiScaleLoss(perceptual_weight=0.0, msssim_weight=0.0)
        gt = torch.rand(1, 3, 32, 32)
        outputs = (
            torch.rand(1, 3, 32, 32),
            torch.rand(1, 3, 16, 16),
            torch.rand(1, 3, 8, 8),
        )

        loss, log_dict = criterion(outputs, gt)

        self.assertEqual(loss.dim(), 0)
        self.assertIn("loss", log_dict)
        self.assertIn("charbonnier", log_dict)

    def test_paired_dataset_returns_aligned_tensors(self):
        dataset_module = importlib.import_module("data.datasets.paired_image_dataset")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder in ["lq", "gt", "lq_s", "gt_s"]:
                (root / folder).mkdir()
                image = np.full((72, 72, 3), 128, dtype=np.uint8)
                cv2.imwrite(str(root / folder / "sample.png"), image)

            dataset = dataset_module.PairedImageDataset(
                root / "lq",
                root / "gt",
                root / "lq_s",
                root / "gt_s",
                phase="train",
                crop_size=64,
                geometric_augs=True,
            )
            sample = dataset[0]

            self.assertEqual(tuple(sample["lq"].shape), (3, 64, 64))
            self.assertEqual(tuple(sample["gt"].shape), (3, 64, 64))
            self.assertEqual(tuple(sample["lq_s"].shape), (3, 64, 64))

    def test_dataloaders_do_not_require_gt_structure_prior(self):
        dataloader_module = importlib.import_module("data.dataloader")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder in ["train_lq", "train_gt", "train_lq_s", "val_lq", "val_gt", "val_lq_s"]:
                (root / folder).mkdir()
                image = np.full((72, 72, 3), 128, dtype=np.uint8)
                cv2.imwrite(str(root / folder / "sample.png"), image)
            config = {
                "training": {"gt_size": 64, "batch_size": 1, "num_workers": 0},
                "validation": {"num_workers": 0},
                "augmentation": {"geometric": False},
            }
            paths = {
                "train_lq_dir": root / "train_lq",
                "train_gt_dir": root / "train_gt",
                "train_lq_s_dir": root / "train_lq_s",
                "val_lq_dir": root / "val_lq",
                "val_gt_dir": root / "val_gt",
                "val_lq_s_dir": root / "val_lq_s",
            }

            train_loader = dataloader_module.build_train_dataloader(config, paths)
            val_loader = dataloader_module.build_val_dataloader(config, paths)

            train_batch = next(iter(train_loader))
            val_batch = next(iter(val_loader))
            self.assertEqual(tuple(train_batch["lq"].shape), (1, 3, 64, 64))
            self.assertEqual(tuple(val_batch["lq"].shape), (1, 3, 72, 72))

    def test_scheduler_steps_without_external_framework(self):
        module = importlib.import_module("utils.scheduler")
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.Adam([parameter], lr=2e-4)
        scheduler = module.CosineAnnealingRestartCyclicLR(
            optimizer,
            periods=[2, 2],
            restart_weights=[1, 1],
            eta_mins=[1e-4, 1e-6],
        )

        optimizer.step()
        scheduler.step()
        optimizer.step()
        scheduler.step()

        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertGreater(optimizer.param_groups[0]["lr"], 0)

    def test_refactor_has_no_external_framework_or_legacy_model_name(self):
        external_name = "basic" + "sr"
        external_title = "Basic" + "SR"
        legacy_name = "".join(chr(v) for v in [85, 72, 68, 77])
        result = subprocess.run(
            ["rg", f"{external_name}|{external_title}|{legacy_name}", str(ROOT)],
            cwd=ROOT.parent,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(result.returncode, 1, msg=result.stdout + result.stderr)

    def test_inference_cli_writes_output_image(self):
        model_module = importlib.import_module("model.sg_llie")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            prior_dir = root / "input_s"
            result_dir = root / "results"
            input_dir.mkdir()
            prior_dir.mkdir()
            image = np.full((64, 64, 3), 96, dtype=np.uint8)
            cv2.imwrite(str(input_dir / "sample.png"), image)
            cv2.imwrite(str(prior_dir / "sample.png"), image)
            config = root / "config.yaml"
            config.write_text(
                "\n".join(
                    [
                        "model:",
                        "  name: SG_LLIE",
                        "  en_feature_num: 8",
                        "  en_inter_num: 4",
                        "  de_feature_num: 8",
                        "  de_inter_num: 4",
                        "  sam_number: 1",
                        "testing:",
                        "  self_ensemble: false",
                        "  factor: 32",
                        "paths:",
                        f"  result_dir: {str(result_dir).replace(chr(92), '/')}",
                    ]
                ),
                encoding="utf-8",
            )
            weights = root / "tiny.pth"
            model = model_module.SG_LLIE(en_feature_num=8, en_inter_num=4, de_feature_num=8, de_inter_num=4, sam_number=1)
            torch.save({"params": model.state_dict(), "model_name": "SG_LLIE"}, weights)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "test" / "test.py"),
                    "--config",
                    str(config),
                    "--input_dir",
                    str(input_dir),
                    "--input_s_dir",
                    str(prior_dir),
                    "--weights",
                    str(weights),
                    "--result_dir",
                    str(result_dir),
                    "--no_self_ensemble",
                ],
                cwd=ROOT.parent,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, msg=completed.stdout + completed.stderr)
            self.assertTrue((result_dir / "sample.png").exists())


if __name__ == "__main__":
    unittest.main()
