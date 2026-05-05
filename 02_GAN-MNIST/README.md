# 02 GAN MNIST — 生成对抗网络手写数字生成

基于 MNIST 数据集的 GAN 图像生成任务，实现并对比 vanilla GAN 与 WGAN-GP 两种生成模型。

## 环境

- Python 3.11 + TensorFlow / Keras
- tensorflow-metal (Apple Silicon GPU)

## 快速开始

```bash
python train_gan.py       # 训练 vanilla GAN
python train_wgan_gp.py   # 训练 WGAN-GP
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `02_GAN-MNIST.ipynb` | 交互式实验 notebook |
| `train_gan.py` | vanilla GAN 训练脚本 |
| `train_wgan_gp.py` | WGAN-GP 训练脚本 |
| `results/` | 训练输出（模型、指标、生成图像） |
| `report.md` | 完整实验报告 |
