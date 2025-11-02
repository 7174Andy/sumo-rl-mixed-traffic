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
