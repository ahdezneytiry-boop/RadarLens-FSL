# YOLO 跌倒识别重跑参考实验

实验日期：2026-06-23

本目录来自第二个重跑版，作为补充参考，不替代主线 EMA-FSL 三阶段训练。主线方案仍以 `02_训练代码/EMA-FSL` 为准。

## 数据来源

数据来自 `YifeiYang210/Fall_Detection_dataset` 中的 Le2i 与 SisFall 处理子集。类别映射如下：

| 原类别 | 本实验类别 |
| --- | --- |
| Fall、Lie、Likefall | 异常 |
| Stand | 正常 |
| Blank | 剔除 |

划分方式为片段级划分，固定随机种子为 `20260623`，同一 16 帧片段内的图片保持在同一划分中，以减少相邻帧泄漏。

## 分类数据规模

| 划分 | 异常图片 | 正常图片 | 总图片 |
| --- | ---: | ---: | ---: |
| 训练集 | 2311 | 1040 | 3351 |
| 验证集 | 494 | 208 | 702 |
| 测试集 | 513 | 240 | 753 |

## 运行环境

| 项目 | 内容 |
| --- | --- |
| 框架 | Ultralytics 8.4.75 |
| Python | 3.12.13 |
| PyTorch | 2.12.1+cpu |
| 设备 | CPU，Intel Core Ultra 9 275HX |

## 分类实验

模型为 YOLO11n-cls，任务为正常和异常二分类。

训练命令：

```powershell
yolo classify train model=yolo11n-cls.pt data=datasets/fall_binary_cls epochs=20 imgsz=224 batch=16 device=cpu workers=0 seed=20260623
```

测试命令：

```powershell
yolo classify val model=runs/fall_yolo_cls/le2i_sisfall_binary/weights/best.pt data=datasets/fall_binary_cls split=test imgsz=224 batch=16 device=cpu workers=0
```

测试集结果：

| 指标 | 数值 |
| --- | ---: |
| 准确率 | 0.8977 |
| 异常类精确率 | 0.8838 |
| 异常类召回率 | 0.9786 |
| 异常类 F1 | 0.9288 |
| 正常类特异度 | 0.7250 |

混淆矩阵：

| 真实类别 | 预测异常 | 预测正常 |
| --- | ---: | ---: |
| 异常 | 502 | 11 |
| 正常 | 66 | 174 |

## 弱框检测实验

该实验将前景轮廓图整张图作为一个弱标注框，不是人工逐帧标注的人体检测框。模型为 YOLO11n。

测试集结果：

| 类别 | 图片数 | 精确率 | 召回率 | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 全部 | 753 | 0.947 | 0.913 | 0.979 | 0.973 |
| 异常 | 513 | 0.935 | 0.981 | 0.989 | 0.987 |
| 正常 | 240 | 0.958 | 0.846 | 0.968 | 0.958 |

## 生成文件

| 文件 | 说明 |
| --- | --- |
| `权重/yolo11n_cls_best.pt` | 分类模型最佳权重 |
| `权重/yolo11n_det_weakbox_best.pt` | 弱框检测模型最佳权重 |
| `图表/cls_confusion_matrix_test.png` | 分类混淆矩阵 |
| `图表/det_confusion_matrix_test.png` | 检测混淆矩阵 |
| `图表/cls_training_curves.png` | 分类训练曲线 |
| `图表/det_training_curves.png` | 检测训练曲线 |

## 使用建议

报告或演示中建议将该结果表述为“YOLO11n 基线与参考实验”。如果展示检测指标，必须说明标注方式为前景轮廓弱框，不能描述成人工标注的人体框数据集。
