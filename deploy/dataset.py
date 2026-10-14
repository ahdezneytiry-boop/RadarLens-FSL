import os
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import pandas as pd

class MultiModalDataset(Dataset):
    def __init__(self, root="data", seq_len=30):
        self.root = root
        self.seq_len = seq_len

        self.labels = pd.read_csv(os.path.join(root, "label.csv"))
        self.ids = self.labels["id"].tolist()

    def __len__(self):
        return len(self.ids)

    def load_radar(self, sample_id):
        path = os.path.join(self.root, "radar", sample_id + ".npy")
        radar = np.load(path)  # shape: (T, d)
        return radar[:self.seq_len]

    def load_visual(self, sample_id):
        folder = os.path.join(self.root, "visual", sample_id)
        frames = sorted(os.listdir(folder))[:self.seq_len]

        imgs = []
        for f in frames:
            img = cv2.imread(os.path.join(folder, f), 0)
            img = cv2.resize(img, (64, 64))
            imgs.append(img)

        imgs = np.stack(imgs)  # (T, H, W)
        imgs = imgs.mean(axis=0, keepdims=True)  # 简化为1帧
        return imgs.astype(np.float32)

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        label = int(self.labels.iloc[idx]["label"])

        radar = self.load_radar(sample_id)
        visual = self.load_visual(sample_id)

        return (
            torch.tensor(radar, dtype=torch.float32),
            torch.tensor(visual, dtype=torch.float32),
            torch.tensor(label)
        )
