import argparse
from pathlib import Path

import torch

from config import Config
from dataset import MultiModalDataset
from model import EMA_FSL


def parse_args():
    parser = argparse.ArgumentParser(description="EMA-FSL 推理入口")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--checkpoint", default=Config.save_path)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--size", type=int, default=16)
    return parser.parse_args()


def inference():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = Path(args.checkpoint)
    data_root = Path(args.data_root)
    synthetic = args.synthetic or not (data_root / "label.csv").exists()

    model = EMA_FSL(Config.num_classes, Config.tau).to(device)
    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        print("已加载模型:", checkpoint)
    elif synthetic:
        print("未找到 checkpoint，使用随机初始化模型进行 synthetic 流程检查。")
    else:
        raise FileNotFoundError(f"未找到模型文件: {checkpoint}")
    model.eval()

    dataset = MultiModalDataset(root=data_root, synthetic=synthetic, size=args.size)
    correct = 0
    with torch.no_grad():
        for radar, visual, label in dataset:
            radar = radar.unsqueeze(0).to(device)
            visual = visual.unsqueeze(0).to(device)
            output = model(radar, visual)
            pred = torch.argmax(output, dim=-1)
            correct += int((pred.cpu() == label).sum().item())

    if synthetic:
        print(f"synthetic smoke 样本数: {len(dataset)}，流程预测匹配率: {correct / len(dataset):.2f}")
        print("说明: synthetic 结果仅用于检查推理链路，不代表真实场景准确率。")
    else:
        print(f"准确率: {correct / len(dataset):.2f}")


if __name__ == "__main__":
    inference()
