# MNIST AutoEncoder — PyTorch 复现与优化

将原 Keras MLP AutoEncoder 迁移到 PyTorch，并通过 CNN + BCE loss 实现更优重建。

## 环境

- Python 3.12 + PyTorch 2.5.1+cu121
- CUDA 12.1 / NVIDIA RTX 4060 Laptop GPU
- 详见 [env_setup.md](env_setup.md)

## 快速开始

```bash
# 1. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 2. 安装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 3. 运行基线模型
python baseline.py          # 输出 baseline_result.png

# 4. 运行优化模型
python improved.py          # 输出 improved_result.png

# 5. 对比两个模型
python compare.py
```

## 模型对比

| 项目 | Baseline | Improved |
|------|----------|----------|
| 结构 | MLP (4层全连接) | CNN (2层Conv + ConvTranspose) |
| 参数量 | 210,562 | 413,315 |
| 损失函数 | MSE | BCE |
| 输出激活 | Tanh | Sigmoid |
| LR 策略 | 固定 1e-3 | ReduceLROnPlateau |
| Epochs | 10 | 20 |
| **Test MSE** | **0.042864** | **0.040677** |

MSE 提升 **5.1%**。

## 文件说明

```
├── baseline.py           # 基线 MLP AutoEncoder (PyTorch 复现)
├── improved.py           # 优化 CNN AutoEncoder
├── compare.py            # 两模型对比脚本
├── baseline_result.png   # 基线可视化输出
├── improved_result.png   # 优化模型可视化输出
├── env_setup.md          # 环境配置说明
├── baseline_result.md    # 基线结果记录
├── improved_result.md    # 优化结果记录
├── requirements.txt      # Python 依赖
├── README.md             # 本文件
└── 03_AutoEncoder-minist.ipynb  # 原始 Keras notebook
```
