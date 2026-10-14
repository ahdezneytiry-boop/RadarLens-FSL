import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import Config
from dataset import MultiModalDataset
from loss import JointLoss
from model import EMA_FSL
from utils import count_parameters

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **_: x


def parse_args():
    parser = argparse.ArgumentParser(description="EMA-FSL 训练入口")
    parser.add_argument("--data-root", default="data", help="真实数据根目录，需包含 label.csv、radar、visual")
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--checkpoint", default=Config.save_path)
    parser.add_argument("--synthetic", action="store_true", help="使用确定性 synthetic smoke 数据")
    parser.add_argument("--size", type=int, default=200, help="synthetic 模式样本数")
    return parser.parse_args()


def train():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root)
    synthetic = args.synthetic or not (data_root / "label.csv").exists()
    if synthetic and not args.synthetic:
        print("未发现真实 label.csv，自动进入 synthetic smoke 模式。")

    dataset = MultiModalDataset(root=data_root, synthetic=synthetic, size=args.size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = EMA_FSL(Config.num_classes, Config.tau).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)
    criterion = JointLoss()

    print(f"device={device}, samples={len(dataset)}, parameters={count_parameters(model)}")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for radar, visual, label in tqdm(dataloader, desc=f"epoch {epoch + 1}/{args.epochs}"):
            radar, visual, label = radar.to(device), visual.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(radar, visual)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / max(1, len(dataloader)):.4f}")

    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    print("模型已保存:", checkpoint)
    if synthetic:
        print("说明: 当前为 synthetic smoke 训练结果，仅用于验证流程，不代表真实数据准确率。")


if __name__ == "__main__":
    train()
