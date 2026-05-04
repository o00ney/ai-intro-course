"""Improved AutoEncoder – CNN + BCE loss + LR scheduler."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
EPOCHS = 20
LR = 1e-3
ENCODING_DIM = 2
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


class ConvAutoEncoder(nn.Module):
    """CNN autoencoder: Conv2d encoder + ConvTranspose2d decoder."""

    def __init__(self, encoding_dim=2):
        super().__init__()
        # Encoder: 1x28x28 → encoding_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),   # 16x14x14
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),  # 32x7x7
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, encoding_dim),
        )

        # Decoder: encoding_dim → 1x28x28
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128), nn.ReLU(),
            nn.Linear(128, 32 * 7 * 7), nn.ReLU(),
            nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),  # 16x14x14
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),  # 1x28x28
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)


def main():
    # --- Data ---
    train_ds = MNIST(root="./data", train=True, download=True, transform=ToTensor())
    test_ds = MNIST(root="./data", train=False, download=True, transform=ToTensor())

    x_train = train_ds.data.numpy().astype("float32") / 255.0
    x_train = x_train.reshape(-1, 1, 28, 28)  # CNN expects [N, C, H, W]

    x_test = test_ds.data.numpy().astype("float32") / 255.0
    x_test = x_test.reshape(-1, 1, 28, 28)
    y_test = test_ds.targets.numpy()

    # Split train into train/val for LR scheduler
    split = int(len(x_train) * 0.9)
    x_tr, x_val = x_train[:split], x_train[split:]
    print(f"Train: {len(x_tr)}, Val: {len(x_val)}, Test: {len(x_test)}")

    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_tr), torch.tensor(x_tr)),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(x_val), torch.tensor(x_val)),
        batch_size=BATCH_SIZE,
    )
    x_test_tensor = torch.tensor(x_test).to(DEVICE)

    # --- Model ---
    model = ConvAutoEncoder(encoding_dim=ENCODING_DIM).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    print(f"Device: {DEVICE}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Encoding dim: {ENCODING_DIM}")
    print()

    # --- Train ---
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                output = model(batch_x)
                val_loss += criterion(output, batch_y).item() * batch_x.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:2d}/{EPOCHS}  train_loss: {epoch_loss:.6f}  val_loss: {val_loss:.6f}  lr: {current_lr:.2e}")

    # --- Evaluate ---
    mse_fn = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        x_flat_test = torch.tensor(test_ds.data.numpy().astype("float32") / 255.0).reshape(-1, 1, 28, 28).to(DEVICE)
        encoded = model.encode(x_flat_test).cpu().numpy()
        final_test_bce = criterion(model(x_test_tensor), x_test_tensor).item()
        final_test_mse = mse_fn(model(x_test_tensor), x_test_tensor).item()

    print(f"\nFinal test BCE: {final_test_bce:.6f}")
    print(f"Final test MSE: {final_test_mse:.6f}")

    torch.save(model.state_dict(), "models/improved_model.pt")
    print("Model saved to models/improved_model.pt")

    # --- Plot ---
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.plot(range(1, EPOCHS + 1), train_losses, "b-o", label="Train")
    plt.plot(range(1, EPOCHS + 1), val_losses, "r-o", label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    scatter = plt.scatter(encoded[:, 0], encoded[:, 1], c=y_test, cmap="tab10", s=2, alpha=0.6)
    plt.colorbar(scatter, ticks=range(10))
    plt.title("Latent Space (CNN AE)")

    plt.subplot(1, 3, 3)
    latent_grid_test(model, DEVICE)
    plt.title("Generated from Latent Grid")

    plt.tight_layout()
    plt.savefig("results/improved_result.png", dpi=100)
    plt.close()

    return {"train_losses": train_losses, "val_losses": val_losses, "final_test_bce": final_test_bce, "final_test_mse": final_test_mse}


def latent_grid_test(model, device, n=20, range_xy=(-3, 3)):
    model.eval()
    fig = np.zeros((28 * n, 28 * n))
    g_x = np.linspace(range_xy[0], range_xy[1], n)
    g_y = np.linspace(range_xy[0], range_xy[1], n)[::-1]
    with torch.no_grad():
        for i, yi in enumerate(g_y):
            for j, xi in enumerate(g_x):
                z = torch.tensor([[xi, yi]], dtype=torch.float32).to(device)
                decoded = model.decoder(z).cpu().numpy()[0, 0]
                fig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = decoded
    plt.imshow(fig, cmap="gray")


if __name__ == "__main__":
    result = main()
    print(f"\nImproved final test BCE: {result['final_test_bce']:.6f}")
    print(f"Improved final test MSE: {result['final_test_mse']:.6f}")
    print(f"Baseline test MSE was: 0.042864")
