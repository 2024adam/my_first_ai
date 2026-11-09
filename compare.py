import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


# ---------- Shared ----------
def novel_loss_function(predictions, targets, model, lambda_complexity=0.001):
    base_loss = nn.MSELoss()(predictions, targets)
    complexity = sum(p.abs().sum() for p in model.parameters())
    return base_loss + lambda_complexity * complexity


def make_dataset(n_samples=2000, seq_len=20, seed=0):
    torch.manual_seed(seed)
    freq = torch.rand(n_samples) * 4 + 0.5
    phase = torch.rand(n_samples) * 6.28
    t = torch.linspace(0, 3, seq_len).unsqueeze(0)
    wave = torch.sin(2 * 3.14159 * freq.unsqueeze(1) * t + phase.unsqueeze(1))
    wave = wave + 0.05 * torch.randn_like(wave)
    targets = torch.stack([freq, phase], dim=1)
    wave = (wave - wave.mean()) / (wave.std() + 1e-6)
    targets = (targets - targets.mean(0)) / (targets.std(0) + 1e-6)
    return wave, targets


def train(model, train_ds, val_ds, epochs=50, lr=1e-3, verbose=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128)
    history = {"train": [], "val": []}

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = novel_loss_function(model(xb), yb, model)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        train_loss = total / len(train_ds)

        model.eval()
        with torch.no_grad():
            val_loss = sum(
                novel_loss_function(model(x.to(device)), y.to(device), model).item()
                for x, y in val_loader
            ) / len(val_loader)

        history["train"].append(train_loss)
        history["val"].append(val_loss)
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:03d}  train {train_loss:.4f}  val {val_loss:.4f}")

    return history


# ---------- V1: fixed-frequency sine interference ----------
class QILayerV1(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(in_features, out_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.phase_shift = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        linear = torch.matmul(x, self.weights) + self.bias
        interference = torch.sin(linear + self.phase_shift)
        return linear * interference


class ModelV1(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super().__init__()
        self.layer1 = QILayerV1(input_dim, hidden_dim)
        self.layer2 = QILayerV1(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        return self.layer3(x)


# ---------- V2: learnable-frequency sine interference ----------
class QILayerV2(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(in_features, out_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.phase_shift = nn.Parameter(torch.randn(out_features))
        self.omega = nn.Parameter(torch.ones(out_features))

    def forward(self, x):
        linear = torch.matmul(x, self.weights) + self.bias
        interference = torch.sin(self.omega * linear + self.phase_shift)
        return linear * interference


class ModelV2(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super().__init__()
        self.layer1 = QILayerV2(input_dim, hidden_dim)
        self.layer2 = QILayerV2(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        return self.layer3(x)


# ---------- Run comparison ----------
if __name__ == "__main__":
    torch.manual_seed(42)
    X, Y = make_dataset(3000)
    n = int(0.8 * len(X))
    train_ds = TensorDataset(X[:n], Y[:n])
    val_ds = TensorDataset(X[n:], Y[n:])

    print("Training V1 (fixed frequency)...")
    m1 = ModelV1(input_dim=20, output_dim=2)
    h1 = train(m1, train_ds, val_ds, epochs=60, lr=1e-3)

    print("Training V2 (learnable omega)...")
    m2 = ModelV2(input_dim=20, output_dim=2)
    h2 = train(m2, train_ds, val_ds, epochs=60, lr=1e-3)

    print()
    print(f"V1 final val loss: {h1['val'][-1]:.4f}")
    print(f"V2 final val loss: {h2['val'][-1]:.4f}")
    print(f"Difference: {h2['val'][-1] - h1['val'][-1]:+.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(h1["val"], label=f"V1 fixed freq  (final {h1['val'][-1]:.4f})")
    plt.plot(h2["val"], label=f"V2 learnable omega  (final {h2['val'][-1]:.4f})")
    plt.xlabel("Epoch")
    plt.ylabel("Validation loss")
    plt.title("V1 vs V2 — sine-interference regression")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("comparison.png", dpi=120)
    print("Saved comparison.png")