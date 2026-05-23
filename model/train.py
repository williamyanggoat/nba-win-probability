import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from model.net import WinProbNet
from model.dataset import GameDataset
import pathlib

CHECKPOINT_DIR = pathlib.Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

def train(
    parquet_path="data/features.parquet",
    epochs=20,
    batch_size=2048,
    lr=1e-3,
    val_split=0.1,
):
    dataset = GameDataset(parquet_path)
    val_n   = int(len(dataset) * val_split)
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_n, val_n])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = WinProbNet().to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(y)

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                val_loss += loss_fn(pred, y).item() * len(y)
                correct  += ((pred > 0.5).float() == y).sum().item()

        train_loss = total_loss / len(train_ds)
        val_loss   = val_loss   / len(val_ds)
        val_acc    = correct    / len(val_ds)
        print(f"Epoch {epoch:02d} | train={train_loss:.4f} | val={val_loss:.4f} | acc={val_acc:.3f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), CHECKPOINT_DIR / "model.pt")
            print("  ✓ checkpoint saved")

    print(f"\nBest val loss: {best_val:.4f}")

if __name__ == "__main__":
    train()