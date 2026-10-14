# RadarLens-FSL · 多模态少样本异常行为识别

![License](https://img.shields.io/github/license/ahdezneytiry-boop/RadarLens-FSL)
![Top Language](https://img.shields.io/github/languages/top/ahdezneytiry-boop/RadarLens-FSL)
![Last Commit](https://img.shields.io/github/last-commit/ahdezneytiry-boop/RadarLens-FSL)
![Repo Size](https://img.shields.io/github/repo-size/ahdezneytiry-boop/RadarLens-FSL)

雷达 + 视觉融合的少样本异常行为识别算法（EMA-FSL），2026 年中国大学生计算机设计大赛参赛作品。

对"行为数据少、异常样本更难收集"的场景，用**多模态融合 + 少样本学习**识别异常行为：雷达信号与视觉图像两条支路联合建模，支持合成数据冒烟测试与真实数据两种模式。

## 目录结构

| 目录/文件 | 说明 |
|---|---|
| `EMA-FSL/` | 算法训练与推理入口（config / dataset / model / loss / train / main） |
| `ONNX-TensorRT/` | ONNX 导出与 TensorRT 部署脚本 |
| `deploy/` | 部署数据组织示例（`label.csv` 为样例数据，无真实数据） |
| `web-ui/` | Flask + HTML5 演示界面（app.py + templates） |
| `YOLO重跑参考实验/` | YOLO 分类/检测重跑实验记录与报告 |
| `demo.py` / `show.py` | 演示脚本 |
| `yolov8n.pt` | YOLOv8n 预训练权重 |

## 快速开始

```bash
pip install -r EMA-FSL/requirements.txt

# 冒烟测试（合成数据，检查全流程）
python EMA-FSL/train.py --synthetic --epochs 1 --checkpoint checkpoints/smoke.pth
python EMA-FSL/main.py --synthetic --checkpoint checkpoints/smoke.pth

# 真实数据：准备 label.csv + radar/*.npy + visual/<id>/*.jpg 后
python EMA-FSL/train.py --data-root data --epochs 10 --checkpoint checkpoints/model.pth
```

## 说明

- 出于隐私与体积考虑，训练数据、演示视频未包含在本仓库。
- 原始作品中的个人署名已替换为 GitHub 用户名。

## 许可证

MIT © 2026 [ahdezneytiry-boop](https://github.com/ahdezneytiry-boop)
