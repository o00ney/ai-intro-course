# Improved Result

## 三项优化

| # | 优化项 | 原方案 | 新方案 | 理由 |
|---|--------|--------|--------|------|
| 1 | 网络结构 | MLP (全连接) | CNN (Conv2d + ConvTranspose2d) | 卷积能捕捉空间局部特征，更适合图像 |
| 2 | 损失函数 | MSE | BCE (Binary Cross Entropy) | BCE 对 [0,1] 归一化像素值更合理，梯度更好 |
| 3 | 学习率调度 | 固定 lr=1e-3 | ReduceLROnPlateau | 自动在 plateau 时降低 lr，提升收敛质量 |

## CNN AutoEncoder 结构
```
Encoder: 1x28x28 → Conv(16,3,2) → Conv(32,3,2) → Flatten → FC(128) → FC(2)
Decoder: FC(2) → FC(128) → FC(1568) → Reshape(32,7,7) → ConvT(16,3,2) → ConvT(1,3,2) → Sigmoid
```
- Parameters: 413,315
- Encoding dim: 2 (保持不变，可做 2D 可视化)

## 训练过程 (20 epochs)
| Epoch | Train BCE | Val BCE | LR    |
|-------|-----------|---------|-------|
| 1     | 0.288106  | 0.238972| 1e-3  |
| 5     | 0.197151  | 0.194587| 1e-3  |
| 10    | 0.187085  | 0.185728| 1e-3  |
| 15    | 0.183338  | 0.182739| 1e-3  |
| 20    | 0.181258  | 0.180784| 1e-3  |

## 对比验证

| Metric | Baseline (MLP+MSE) | Improved (CNN+BCE) | 提升 |
|--------|-------------------|--------------------|------|
| MSE    | 0.042864          | 0.040677           | **5.1%** |
| BCE    | 0.721168          | 0.181863           | **74.8%** |

Improved MSE (0.040677) < Baseline MSE (0.042864): **PASS**
