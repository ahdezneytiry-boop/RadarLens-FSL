import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1] / "EMA-FSL"
sys.path.insert(0, str(ROOT))

from model import EMA_FSL


def parse_args():
    parser = argparse.ArgumentParser(description="导出 EMA-FSL ONNX 模型")
    parser.add_argument("--checkpoint", default="../EMA-FSL/checkpoints/model.pth")
    parser.add_argument("--output", default="ema_fsl.onnx")
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main():
    args = parse_args()
    model = EMA_FSL(num_classes=2, tau=0.7)
    checkpoint = Path(args.checkpoint)
    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        print("已加载 checkpoint:", checkpoint)
    else:
        print("未找到 checkpoint，将导出随机初始化模型用于部署链路检查:", checkpoint)
    model.eval()
    radar = torch.randn(1, 30, 16)
    visual = torch.randn(1, 1, 64, 64)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (radar, visual),
        output,
        input_names=["radar", "visual"],
        output_names=["logits"],
        opset_version=args.opset,
        dynamic_axes={"radar": {0: "batch"}, "visual": {0: "batch"}, "logits": {0: "batch"}},
    )
    print("ONNX 已导出:", output)


if __name__ == "__main__":
    main()
