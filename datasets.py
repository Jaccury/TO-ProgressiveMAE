# datasets.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class RGBDTransparentDataset(Dataset):
    """
    通用 ClearGrasp / Trans10K RGB-D 数据集：
    假设目录结构：
        base_root/
            ClearGrasp/
                train/
                    rgb/
                    depth/
                val/
                    rgb/
                    depth/
            Trans10K/
                train/
                    rgb/
                    depth/
                val/
                    rgb/
                    depth/

    深度可以是 .npy 或 16bit png。
    """

    def __init__(self,
                 base_root,
                 dataset_name="ClearGrasp",
                 split="train",
                 transform=None,
                 depth_scale=1.0):
        """
        base_root: r"E:\论文\我的\深度补全预训练模型\pythonProject1\databases"
        dataset_name: "ClearGrasp" or "Trans10K"
        split: "train" or "val"
        """
        self.base_root = base_root
        self.dataset_name = dataset_name
        self.split = split
        self.transform = transform
        self.depth_scale = depth_scale

        self.rgb_dir = os.path.join(base_root, dataset_name, split, "rgb")
        self.depth_dir = os.path.join(base_root, dataset_name, split, "depth")

        assert os.path.isdir(self.rgb_dir), f"RGB dir not found: {self.rgb_dir}"
        assert os.path.isdir(self.depth_dir), f"Depth dir not found: {self.depth_dir}"

        self.ids = sorted([
            os.path.splitext(f)[0]
            for f in os.listdir(self.rgb_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        assert len(self.ids) > 0, f"No images found in {self.rgb_dir}"

    def __len__(self):
        return len(self.ids)

    def _load_depth(self, path):
        if path.lower().endswith(".npy"):
            depth = np.load(path).astype(np.float32)
        else:
            # 假设是 16bit PNG
            d = Image.open(path)
            depth = np.array(d).astype(np.float32) / 1000.0  # 按需调整
        return depth

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        # RGB
        rgb_path_png = os.path.join(self.rgb_dir, id_ + ".png")
        rgb_path_jpg = os.path.join(self.rgb_dir, id_ + ".jpg")
        if os.path.exists(rgb_path_png):
            rgb_path = rgb_path_png
        else:
            rgb_path = rgb_path_jpg

        # Depth
        depth_path_npy = os.path.join(self.depth_dir, id_ + ".npy")
        depth_path_png = os.path.join(self.depth_dir, id_ + ".png")
        if os.path.exists(depth_path_npy):
            depth = self._load_depth(depth_path_npy)
        elif os.path.exists(depth_path_png):
            depth = self._load_depth(depth_path_png)
        else:
            raise FileNotFoundError(f"Depth file not found for id {id_}")

        rgb = Image.open(rgb_path).convert("RGB")
        rgb = np.array(rgb).astype(np.float32) / 255.0
        depth = depth * self.depth_scale

        # 有效像素 mask（>0）
        valid_mask = (depth > 0).astype(np.float32)

        # H,W,C -> C,H,W
        rgb = torch.from_numpy(rgb).permute(2, 0, 1)  # (3,H,W)
        depth = torch.from_numpy(depth).unsqueeze(0)  # (1,H,W)
        valid_mask = torch.from_numpy(valid_mask).unsqueeze(0)  # (1,H,W)

        if self.transform is not None:
            # 如果你有统一的几何变换，可以写在这里（保证 RGB/Depth 同步）
            pass

        return {
            "id": id_,
            "rgb": rgb,
            "depth": depth,
            "valid_mask": valid_mask
        }
