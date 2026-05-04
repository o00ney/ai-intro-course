"""Head-to-head comparison: baseline MLP AE vs improved CNN AE."""
import torch
import torch.nn as nn
import numpy as np
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Re-define both models (same classes as baseline.py / improved.py)


class MLP_AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 128), nn.ReLU(),
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, 8), nn.ReLU(),
            nn.Linear(8, 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 8), nn.ReLU(),
            nn.Linear(8, 32), nn.ReLU(),
            nn.Linear(32, 128), nn.ReLU(),
            nn.Linear(128, 784), nn.Tanh(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class CNN_AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 128), nn.ReLU(),
            nn.Linear(128, 32 * 7 * 7), nn.ReLU(),
            nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def load_data():
    test_ds = MNIST(root="./data", train=False, download=True, transform=ToTensor())
    x_flat = torch.tensor(test_ds.data.numpy().astype("float32") / 255.0).reshape(-1, 784).to(DEVICE)
    x_cnn = torch.tensor(test_ds.data.numpy().astype("float32") / 255.0).reshape(-1, 1, 28, 28).to(DEVICE)
    return x_flat, x_cnn


def main():
    x_flat, x_cnn = load_data()

    # Load trained weights
    baseline_ckpt = torch.load("models/baseline_model.pt", map_location=DEVICE, weights_only=True)
    improved_ckpt = torch.load("models/improved_model.pt", map_location=DEVICE, weights_only=True)

    mlp = MLP_AutoEncoder().to(DEVICE)
    mlp.load_state_dict(baseline_ckpt)
    mlp.eval()
    cnn = CNN_AutoEncoder().to(DEVICE)
    cnn.load_state_dict(improved_ckpt)
    cnn.eval()

    mse = nn.MSELoss()
    bce = nn.BCELoss()

    with torch.no_grad():
        mlp_out = mlp(x_flat)                        # [-1, 1] (tanh)
        cnn_out = cnn(x_cnn)                         # [0, 1] (sigmoid)
        # Normalize MLP tanh output from [-1,1] to [0,1] for BCE comparison
        mlp_out_norm = (mlp_out + 1) / 2

        mlp_mse = mse(mlp_out, x_flat).item()
        mlp_bce = bce(mlp_out_norm, x_flat).item()
        cnn_mse = mse(cnn_out, x_cnn).item()
        cnn_bce = bce(cnn_out, x_cnn).item()

    print("=" * 55)
    print(f"{'Metric':<12} {'Baseline (MLP)':<20} {'Improved (CNN)':<20}")
    print("=" * 55)
    print(f"{'MSE':<12} {mlp_mse:<20.6f} {cnn_mse:<20.6f}")
    print(f"{'BCE':<12} {mlp_bce:<20.6f} {cnn_bce:<20.6f}")
    print("=" * 55)

    mse_improve = (mlp_mse - cnn_mse) / mlp_mse * 100
    print(f"\nMSE improvement: {mse_improve:.1f}%")
    print(f"Optimized MSE ({cnn_mse:.6f}) < Baseline MSE ({mlp_mse:.6f}): {'PASS' if cnn_mse < mlp_mse else 'FAIL'}")

    # Save comparison data
    np.savez("models/comparison.npz",
             baseline_mse=mlp_mse, improved_mse=cnn_mse,
             baseline_bce=mlp_bce, improved_bce=cnn_bce,
             mse_improvement_pct=mse_improve)


if __name__ == "__main__":
    main()
