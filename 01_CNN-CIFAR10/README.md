# 01 CNN CIFAR-10 — 卷积神经网络图像分类与优化

基于 CIFAR-10 数据集的 CNN 图像分类任务，探索数据增强、正则化、学习率调度与早停机制对模型性能的影响。

## 环境

- Python 3.11 + TensorFlow / Keras
- tensorflow-metal (Apple Silicon GPU)

## 快速开始

```bash
python train.py
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `01_CNN-CIFAR10.ipynb` | 交互式实验 notebook |
| `train.py` | 独立训练脚本 |
| `cifar-10-python/` | CIFAR-10 数据集 |
| `results/` | 训练输出（模型、指标、曲线） |
| `report.md` | 完整实验报告 |
