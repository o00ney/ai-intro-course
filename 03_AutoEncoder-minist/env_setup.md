# Environment Setup

## System
- **OS**: Ubuntu 22.04 / WSL2 (Linux 6.6.87)
- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB)
- **CUDA**: 12.1 (Driver 13.0)

## Python Environment
- **Python**: 3.12.3
- **Virtual Environment**: `venv/` at project root

## Key Packages
| Package     | Version      |
|-------------|--------------|
| torch       | 2.5.1+cu121  |
| torchvision | 0.20.1+cu121 |
| torchaudio  | 2.5.1+cu121  |
| numpy       | 2.4.3        |
| matplotlib  | 3.10.9       |
| scikit-learn| 1.8.0        |
| cuDNN       | 9.1.0        |

## GPU Verification
```
PyTorch 2.5.1+cu121
CUDA available: True
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
cuDNN: 90100
```

## Setup Commands
```bash
sudo apt-get install python3.12-venv python3-full
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install matplotlib scikit-learn
```
