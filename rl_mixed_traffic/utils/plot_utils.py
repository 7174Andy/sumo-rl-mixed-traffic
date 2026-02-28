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

def plot_cav_spacing(
    spacings: dict[str, list],
    out_path: str,
    title: str = "CAV Spacing Over Time",
):
    """Plot CAV gap-to-leader (bumper-to-bumper) over evaluation steps.

    Args:
        spacings: Dict mapping agent_id -> list of gap values (m) per step.
        out_path: Path to save the plot.
        title: Title for the plot.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    for aid, vals in spacings.items():
        plt.plot(range(len(vals)), vals, label=aid, linewidth=1.5, alpha=0.8)
    plt.xlabel("Step")
    plt.ylabel("Spacing (m)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[OK] Saved plot: {out_path}")


def plot_accelerations(
    head_accels: list,
    cav_accels: dict[str, list],
    out_path: str,
    title: str = "Vehicle Accelerations Over Time",
):
    """Plot head vehicle and CAV accelerations over evaluation steps.

    Args:
        head_accels: List of head vehicle accelerations (m/s^2) per step.
        cav_accels: Dict mapping agent_id -> list of accel values per step.
        out_path: Path to save the plot.
        title: Title for the plot.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(
        range(len(head_accels)), head_accels,
        label="Head Vehicle", linewidth=1.5, alpha=0.8,
    )
    for aid, vals in cav_accels.items():
        plt.plot(range(len(vals)), vals, label=f"CAV ({aid})", linewidth=1.5, alpha=0.8)
    plt.xlabel("Step")
    plt.ylabel("Acceleration (m/s²)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[OK] Saved plot: {out_path}")


def plot_ppo_metrics(metrics_history: dict, out_dir: str = "ppo_results"):
    """Plot PPO training metrics.

    When Lagrangian keys (lambda, mean_violation, safety_clip_rate) are present,
    uses a 3x2 grid instead of 2x2 to show the additional metrics.

    Args:
        metrics_history: Dictionary of metric names to lists of values
        out_dir: Output directory for plots
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    lagrangian_keys = {"lambda", "mean_violation", "safety_clip_rate"}
    has_lagrangian = any(
        k in metrics_history and len(metrics_history[k]) > 0
        for k in lagrangian_keys
    )

    if has_lagrangian:
        fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    fig.suptitle("PPO Training Metrics", fontsize=16)

    # Policy Loss
    if "policy_loss" in metrics_history and len(metrics_history["policy_loss"]) > 0:
        axes[0, 0].plot(metrics_history["policy_loss"])
        axes[0, 0].set_title("Policy Loss")
        axes[0, 0].set_xlabel("Update")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].grid(True)

    # Value Loss
    if "value_loss" in metrics_history and len(metrics_history["value_loss"]) > 0:
        axes[0, 1].plot(metrics_history["value_loss"])
        axes[0, 1].set_title("Value Loss")
        axes[0, 1].set_xlabel("Update")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].grid(True)

    # Entropy
    if "entropy" in metrics_history and len(metrics_history["entropy"]) > 0:
        axes[1, 0].plot(metrics_history["entropy"])
        axes[1, 0].set_title("Policy Entropy")
        axes[1, 0].set_xlabel("Update")
        axes[1, 0].set_ylabel("Entropy")
        axes[1, 0].grid(True)

    # Clip Fraction
    if "clipfrac" in metrics_history and len(metrics_history["clipfrac"]) > 0:
        axes[1, 1].plot(metrics_history["clipfrac"])
        axes[1, 1].set_title("Clip Fraction")
        axes[1, 1].set_xlabel("Update")
        axes[1, 1].set_ylabel("Fraction")
        axes[1, 1].grid(True)

    # Lagrangian-specific metrics (row 3)
    if has_lagrangian:
        if "lambda" in metrics_history and len(metrics_history["lambda"]) > 0:
            axes[2, 0].plot(metrics_history["lambda"], color="purple")
            axes[2, 0].set_title("Lagrange Multiplier (lambda)")
            axes[2, 0].set_xlabel("Update")
            axes[2, 0].set_ylabel("Lambda")
            axes[2, 0].grid(True)

        # Combine violation and clip rate on the same subplot
        ax_right = axes[2, 1]
        plotted = False
        if "mean_violation" in metrics_history and len(metrics_history["mean_violation"]) > 0:
            ax_right.plot(
                metrics_history["mean_violation"],
                color="red", label="Mean Violation (m)",
            )
            plotted = True
        if "safety_clip_rate" in metrics_history and len(metrics_history["safety_clip_rate"]) > 0:
            ax_twin = ax_right.twinx()
            ax_twin.plot(
                metrics_history["safety_clip_rate"],
                color="orange", label="Safety Clip Rate",
            )
            ax_twin.set_ylabel("Clip Rate")
            ax_twin.legend(loc="upper right")
        if plotted:
            ax_right.set_title("Violation & Safety Clip Rate")
            ax_right.set_xlabel("Update")
            ax_right.set_ylabel("Violation (m)")
            ax_right.legend(loc="upper left")
            ax_right.grid(True)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/ppo_training_metrics.png", dpi=150)
    plt.close()