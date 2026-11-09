import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


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


class QuantumInterferenceLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(in_features, out_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.phase_shift = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        linear = torch.matmul(x, self.weights) + self.bias
        interference = torch.sin(linear + self.phase_shift)
        return linear * interference


class NeverSeenBeforeAI(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super().__init__()
        self.layer1 = QuantumInterferenceLayer(input_dim, hidden_dim)
        self.layer2 = QuantumInterferenceLayer(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        return self.layer3(x)


def train(model, train_ds, val_ds, epochs=50, lr=1e-3):
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
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:03d}  train {train_loss:.4f}  val {val_loss:.4f}")

    return history


if __name__ == "__main__":
    X, Y = make_dataset(3000)
    n = int(0.8 * len(X))
    train_ds = TensorDataset(X[:n], Y[:n])
    val_ds = TensorDataset(X[n:], Y[n:])

    model = NeverSeenBeforeAI(input_dim=20, output_dim=2)
    history = train(model, train_ds, val_ds, epochs=60, lr=1e-3)

    plt.plot(history["train"], label="train")
    plt.plot(history["val"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig("loss_curve.png")
    print("Saved loss_curve.png")