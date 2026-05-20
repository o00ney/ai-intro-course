# 04 UNet++ — 医学细胞核分割

基于 UNet++ (NestedUNet without deep supervision) 的 DSB2018 细胞核图像语义分割任务。

## 环境

- Python 3.12 + PyTorch 2.5.1
- CUDA 13.0 (NVIDIA RTX 4060 Laptop GPU)

## 快速开始

```bash
source ../venv/bin/activate
python train.py
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `train.py` | 训练脚本（含模型定义、数据加载、训练与验证） |
| `inputs/` | 预处理后的 DSB2018 细胞核数据集 (96x96) |
| `results/` | 训练输出（模型权重、指标、曲线） |
| `report.md` | 完整实验报告 |
