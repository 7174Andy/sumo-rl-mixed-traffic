"""Train NNMPC: supervised learning on DeeP-LCC dataset.

Usage:
    uv run rl_mixed_traffic/deep_lcc/nnmpc_train.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from rl_mixed_traffic.deep_lcc.nnmpc_config import NNMPCConfig
from rl_mixed_traffic.deep_lcc.nnmpc_network import NNMPCNetwork


def load_dataset(
    path: str, val_split: float, seed: int = 42
) -> tuple[TensorDataset, TensorDataset, np.ndarray, np.ndarray]:
    """Load dataset and split into train/val.

    Returns:
        (train_dataset, val_dataset, input_mean, input_std)
    """
    data = np.load(path)
    X = np.hstack([data["uini"], data["yini"], data["eini"]]).astype(np.float32)
    y = data["u_opt"].astype(np.float32)

    # Shuffle
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    # Split
    n_val = int(len(X) * val_split)
    X_train, X_val = X[n_val:], X[:n_val]
    y_train, y_val = y[n_val:], y[:n_val]

    # Normalize inputs (fit on train only)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-8] = 1.0  # avoid division by zero for constant features
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    return train_ds, val_ds, mean, std


def train(config: NNMPCConfig | None = None) -> None:
    if config is None:
        config = NNMPCConfig()

    device = torch.device(config.device)
    print(f"Device: {device}")

    # Load data
    train_ds, val_ds, input_mean, input_std = load_dataset(
        config.dataset_path, config.val_split
    )
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)

    input_dim = train_ds[0][0].shape[0]
    output_dim = train_ds[0][1].shape[0]
    print(f"Input dim: {input_dim}, Output dim: {output_dim}")
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Model
    model = NNMPCNetwork(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=config.hidden_dims,
        accel_min=config.accel_min,
        accel_max=config.accel_max,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    loss_fn = nn.MSELoss()

    # Training loop with early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(config.num_epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / n_batches

        # Validate
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                val_loss += loss_fn(pred, y_batch).item()
                n_val += 1
        val_loss /= n_val

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % 10 == 0 or val_loss < best_val_loss:
            print(
                f"Epoch {epoch:3d}/{config.num_epochs} | "
                f"Train: {train_loss:.6f} | Val: {val_loss:.6f}"
                + (" *" if val_loss < best_val_loss else "")
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best checkpoint
            out_path = Path(config.model_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_mean": input_mean,
                    "input_std": input_std,
                    "input_dim": input_dim,
                    "output_dim": output_dim,
                    "config": {
                        "hidden_dims": config.hidden_dims,
                        "accel_min": config.accel_min,
                        "accel_max": config.accel_max,
                    },
                },
                out_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"Early stopping at epoch {epoch} (patience={config.patience})")
                break

    print(f"\nBest val loss: {best_val_loss:.6f}")
    print(f"Model saved to {config.model_path}")

    # Plot loss curves
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label="Train")
    ax.plot(val_losses, label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("NNMPC Training")
    fig_path = Path(config.model_path).parent / "nnmpc_training_loss.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Loss plot saved to {fig_path}")


if __name__ == "__main__":
    train()
