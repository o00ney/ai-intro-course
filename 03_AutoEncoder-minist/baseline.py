"""Baseline AutoEncoder – PyTorch port of the original Keras notebook."""
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
EPOCHS = 10
LR = 1e-3
ENCODING_DIM = 2
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


class AutoEncoder(nn.Module):
    def __init__(self, encoding_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 128), nn.ReLU(),
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, 8), nn.ReLU(),
            nn.Linear(8, encoding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 8), nn.ReLU(),
            nn.Linear(8, 32), nn.ReLU(),
            nn.Linear(32, 128), nn.ReLU(),
            nn.Linear(128, 784), nn.Tanh(),
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
    x_train = x_train.reshape(-1, 784)

    x_test = test_ds.data.numpy().astype("float32") / 255.0
    x_test = x_test.reshape(-1, 784)
    y_test = test_ds.targets.numpy()

    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train), torch.tensor(x_train)),
        batch_size=BATCH_SIZE, shuffle=True,
    )

    # --- Model ---
    model = AutoEncoder(encoding_dim=ENCODING_DIM).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"Device: {DEVICE}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Encoding dim: {ENCODING_DIM}")
    print()

    # --- Train ---
    losses = []
    model.train()
    for epoch in range(1, EPOCHS + 1):
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
        losses.append(epoch_loss)
        print(f"Epoch {epoch:2d}/{EPOCHS}  loss: {epoch_loss:.6f}")

    # --- Encode test set for visualization ---
    model.eval()
    with torch.no_grad():
        x_test_tensor = torch.tensor(x_test).to(DEVICE)
        encoded = model.encode(x_test_tensor).cpu().numpy()
        final_loss = criterion(
            model(x_test_tensor), x_test_tensor
        ).item()

    print(f"\nFinal test MSE: {final_loss:.6f}")

    torch.save(model.state_dict(), "baseline_model.pt")
    print("Model saved to baseline_model.pt")

    # --- Plot ---
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.plot(range(1, EPOCHS + 1), losses, "b-o")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    scatter = plt.scatter(encoded[:, 0], encoded[:, 1], c=y_test, cmap="tab10", s=2, alpha=0.6)
    plt.colorbar(scatter, ticks=range(10))
    plt.title("Latent Space (test set)")

    plt.subplot(1, 3, 3)
    latent_grid_test(model, DEVICE)
    plt.title("Generated from Latent Grid")

    plt.tight_layout()
    plt.savefig("baseline_result.png", dpi=100)
    plt.close()

    return {"train_losses": losses, "final_test_mse": final_loss}


def latent_grid_test(model, device, n=20, range_xy=(-3, 3)):
    model.eval()
    fig = np.zeros((28 * n, 28 * n))
    g_x = np.linspace(range_xy[0], range_xy[1], n)
    g_y = np.linspace(range_xy[0], range_xy[1], n)[::-1]
    with torch.no_grad():
        for i, yi in enumerate(g_y):
            for j, xi in enumerate(g_x):
                z = torch.tensor([[xi, yi]], dtype=torch.float32).to(device)
                decoded = model.decoder(z).cpu().numpy()[0].reshape(28, 28)
                fig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = decoded
    plt.imshow(fig, cmap="gray")


if __name__ == "__main__":
    result = main()
    print(f"\nBaseline final test MSE: {result['final_test_mse']:.6f}")
