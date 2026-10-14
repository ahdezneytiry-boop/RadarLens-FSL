import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class MultiModalDataset(Dataset):
    def __init__(self, root="data", seq_len=30, img_size=64, synthetic=True, size=200, seed=2026):
        self.root = Path(root)
        self.seq_len = seq_len
        self.img_size = img_size
        self.synthetic = synthetic
        self.size = size
        self.seed = seed
        self.items = []

        if not synthetic:
            label_path = self.root / "label.csv"
            if not label_path.exists():
                raise FileNotFoundError(f"真实数据模式需要标签文件: {label_path}")
            with label_path.open("r", encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    self.items.append({"id": row["id"], "label": int(row["label"])})
            if not self.items:
                raise ValueError("label.csv 中没有可用样本")

    def __len__(self):
        return self.size if self.synthetic else len(self.items)

    def __getitem__(self, idx):
        if self.synthetic:
            return self._synthetic_item(idx)
        item = self.items[idx]
        radar = self._load_radar(item["id"])
        visual = self._load_visual(item["id"])
        return torch.tensor(radar), torch.tensor(visual), torch.tensor(item["label"], dtype=torch.long)

    def _synthetic_item(self, idx):
        rng = np.random.default_rng(self.seed + idx)
        label = idx % 2
        radar = rng.normal(0, 0.08, size=(self.seq_len, 16)).astype(np.float32)
        visual = rng.normal(0, 0.03, size=(1, self.img_size, self.img_size)).astype(np.float32)
        if label == 1:
            center = self.seq_len // 2
            radar[center:center + 4, 1] += np.array([0.55, 1.05, 1.45, 0.65], dtype=np.float32)
            radar[center:center + 4, 2] += np.array([0.22, 0.46, 0.82, 0.28], dtype=np.float32)
            visual[:, self.img_size // 2:, self.img_size // 3: self.img_size // 3 * 2] += 0.55
        return torch.tensor(radar), torch.tensor(visual), torch.tensor(label, dtype=torch.long)

    def _load_radar(self, sample_id):
        path = self.root / "radar" / f"{sample_id}.npy"
        if not path.exists():
            raise FileNotFoundError(f"缺少雷达样本: {path}")
        radar = np.load(path).astype(np.float32)
        radar = self._fit_time(radar, self.seq_len)
        if radar.shape[1] < 16:
            radar = np.pad(radar, ((0, 0), (0, 16 - radar.shape[1])), mode="constant")
        return radar[:, :16]

    def _load_visual(self, sample_id):
        try:
            import cv2
        except Exception as exc:
            raise RuntimeError("真实视觉数据读取需要安装 opencv-python") from exc

        folder = self.root / "visual" / sample_id
        if not folder.exists():
            raise FileNotFoundError(f"缺少视觉样本目录: {folder}")
        frames = sorted([p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])[:self.seq_len]
        if not frames:
            raise ValueError(f"视觉样本目录为空: {folder}")
        imgs = []
        for frame in frames:
            img = cv2.imread(str(frame), 0)
            if img is None:
                continue
            img = cv2.resize(img, (self.img_size, self.img_size))
            imgs.append(img.astype(np.float32) / 255.0)
        if not imgs:
            raise ValueError(f"视觉样本不可读取: {folder}")
        return np.stack(imgs).mean(axis=0, keepdims=True).astype(np.float32)

    def _fit_time(self, arr, length):
        if arr.ndim != 2:
            raise ValueError("雷达样本应为二维数组: (T, D)")
        if arr.shape[0] >= length:
            return arr[:length]
        pad = np.zeros((length - arr.shape[0], arr.shape[1]), dtype=arr.dtype)
        return np.concatenate([arr, pad], axis=0)
