import argparse
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="EMA-FSL ONNX 推理检查")
    parser.add_argument("--model", default="ema_fsl.onnx")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        import onnxruntime as ort
    except Exception:
        print("未安装 onnxruntime。请先执行: pip install onnxruntime")
        return

    model_path = Path(args.model)
    if not model_path.exists():
        print("未找到 ONNX 模型，请先运行 export_onnx.py:", model_path)
        return

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    feeds = {}
    for item in session.get_inputs():
        if item.name == "radar":
            feeds[item.name] = np.random.randn(1, 30, 16).astype(np.float32)
        elif item.name == "visual":
            feeds[item.name] = np.random.randn(1, 1, 64, 64).astype(np.float32)
    outputs = session.run(None, feeds)
    print("ONNX 推理完成，输出形状:", [x.shape for x in outputs])


if __name__ == "__main__":
    main()
