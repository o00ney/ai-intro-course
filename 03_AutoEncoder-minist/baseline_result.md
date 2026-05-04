# Baseline Result

## Model Architecture (MLP AutoEncoder)
```
Encoder: 784 → 128 → 32 → 8 → 2 (encoding_dim)
Decoder: 2 → 8 → 32 → 128 → 784 → tanh
```
- Parameters: 210,562
- Loss: MSE
- Optimizer: Adam (lr=1e-3)
- Epochs: 10
- Batch size: 256

## Training Results
| Epoch | Train Loss |
|-------|-----------|
| 1     | 0.064943  |
| 2     | 0.055258  |
| 3     | 0.050410  |
| 4     | 0.047880  |
| 5     | 0.046219  |
| 6     | 0.045199  |
| 7     | 0.044496  |
| 8     | 0.043928  |
| 9     | 0.043484  |
| 10    | 0.043114  |

## Final Result
- **Train Loss (final epoch):** 0.043114
- **Test MSE:** 0.042864
- **Device:** CUDA (NVIDIA GeForce RTX 4060 Laptop GPU)
