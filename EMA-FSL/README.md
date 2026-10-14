# EMA-FSL 多模态异常行为识别算法

本目录保留算法训练与推理入口。默认支持两种模式：

- `synthetic smoke`：使用确定性模拟数据检查训练、保存、推理流程。
- `真实数据`：读取 `label.csv`、`radar/*.npy`、`visual/<sample_id>/*.jpg`。

## 环境

```bash
pip install -r requirements.txt
```

## 训练

```bash
python train.py --synthetic --epochs 1 --checkpoint checkpoints/smoke.pth
```

真实数据：

```bash
python train.py --data-root data --epochs 10 --checkpoint checkpoints/model.pth
```

## 推理

```bash
python main.py --synthetic --checkpoint checkpoints/smoke.pth
```

如果没有真实 `label.csv`，程序会自动进入 synthetic smoke 模式，并明确提示该结果不代表真实准确率。
