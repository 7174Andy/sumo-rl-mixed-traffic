import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def moving_average(x, w=10):
    x = np.asarray(x, dtype=float)
    if w <= 1 or len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def plot_returns(
    returns, out_path: str, smooth_window: int = 10, title: str = "Episode Returns"
):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    # raw
    plt.plot(range(1, len(returns) + 1), returns, label="Return")
    # smoothed
    ma = moving_average(returns, smooth_window)
    if len(ma) > 1 and len(returns) >= smooth_window:
        plt.plot(
            range(smooth_window, len(returns) + 1), ma, label=f"MA({smooth_window})"
        )
    plt.xlabel("Episode")
    plt.ylabel("Sum of rewards")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[OK] Saved plot: {out_path}")


def plot_losses(losses, out_path: str, title: str = "Training Losses"):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    # raw
    plt.plot(range(1, len(losses) + 1), losses, label="Loss")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[OK] Saved plot: {out_path}")


def plot_vehicle_speeds(
    head_speeds, cav_speeds, out_path: str, title: str = "Vehicle Speeds Over Time"
):
    """Plot the speeds of the head vehicle and CAV over evaluation steps.

    Args:
        head_speeds: List of head vehicle speeds (m/s) at each step
        cav_speeds: List of CAV (agent) vehicle speeds (m/s) at each step
        out_path: Path to save the plot
        title: Title for the plot
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    steps = range(len(cav_speeds))

    plt.figure(figsize=(10, 5))
    plt.plot(steps, head_speeds, label="Head Vehicle", linewidth=1.5, alpha=0.8)
    plt.plot(steps, cav_speeds, label="CAV (Agent)", linewidth=1.5, alpha=0.8)
    plt.xlabel("Step")
    plt.ylabel("Speed (m/s)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[OK] Saved plot: {out_path}")
