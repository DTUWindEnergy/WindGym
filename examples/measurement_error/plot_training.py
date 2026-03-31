"""Parse a protagonist training log (SB3 format) and plot progress.

Usage:
    python plot_training.py <log_file> [--output <output_path>]
"""

import re
import sys
import argparse
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_log(log_file):
    iterations = []
    current = {}

    with open(log_file) as f:
        for line in f:
            line = line.strip()
            # Match lines like: |    ep_rew_mean     | 47.5     |
            m = re.match(r"\|\s+([\w/]+)\s+\|\s+([\d.e+\-]+)\s+\|", line)
            if m:
                key = m.group(1).strip()
                val = float(m.group(2))
                current[key] = val
            # A separator line after a block means the block is done
            elif line.startswith("---") and current:
                iterations.append(current.copy())
                current = {}

    return iterations


def main():
    parser = argparse.ArgumentParser(
        description="Plot SB3 training progress from log file"
    )
    parser.add_argument("log_file", help="Path to the SB3 training log file")
    parser.add_argument(
        "--output", default="training_progress.png", help="Output plot filename"
    )
    args = parser.parse_args()

    iterations = parse_log(args.log_file)
    print(f"Parsed {len(iterations)} logging blocks from {args.log_file}")

    if not iterations:
        print(
            "No logging blocks found. Check that the log file contains SB3-formatted output."
        )
        sys.exit(1)

    # Extract metrics into arrays
    def extract(key):
        return [it[key] for it in iterations if key in it]

    timesteps = extract("total_timesteps")
    ep_rew = extract("ep_rew_mean")
    ep_len = extract("ep_len_mean")

    # train/ metrics start from iteration 2
    train_iters = [it for it in iterations if "value_loss" in it]
    train_ts = [it["total_timesteps"] for it in train_iters]
    value_loss = [it["value_loss"] for it in train_iters]
    pg_loss = [it["policy_gradient_loss"] for it in train_iters]
    entropy = [it["entropy_loss"] for it in train_iters]
    expl_var = [it["explained_variance"] for it in train_iters]
    std = [it["std"] for it in train_iters]

    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(
        f"Training Progress\n({args.log_file})", fontsize=14, fontweight="bold"
    )

    ts_k = [t / 1000 for t in timesteps]
    train_ts_k = [t / 1000 for t in train_ts]

    # 1. Episode reward (most important)
    ax = axes[0, 0]
    ax.plot(ts_k, ep_rew, color="tab:blue", linewidth=1.5)
    ax.set_title("Episode Reward Mean")
    ax.set_xlabel("Timesteps (k)")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.3)

    # 2. Episode length
    ax = axes[0, 1]
    ax.plot(ts_k, ep_len, color="tab:orange", linewidth=1.5)
    ax.set_title("Episode Length Mean")
    ax.set_xlabel("Timesteps (k)")
    ax.set_ylabel("Steps")
    ax.grid(True, alpha=0.3)

    # 3. Value loss
    ax = axes[1, 0]
    ax.plot(train_ts_k, value_loss, color="tab:red", linewidth=1.5)
    ax.set_title("Value Loss")
    ax.set_xlabel("Timesteps (k)")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    # 4. Policy gradient loss
    ax = axes[1, 1]
    ax.plot(train_ts_k, pg_loss, color="tab:green", linewidth=1.5)
    ax.set_title("Policy Gradient Loss")
    ax.set_xlabel("Timesteps (k)")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    # 5. Explained variance
    ax = axes[2, 0]
    ax.plot(train_ts_k, expl_var, color="tab:purple", linewidth=1.5)
    ax.set_title("Explained Variance")
    ax.set_xlabel("Timesteps (k)")
    ax.set_ylabel("Explained Var")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # 6. Entropy loss & policy std
    ax = axes[2, 1]
    ax2 = ax.twinx()
    (l1,) = ax.plot(
        train_ts_k,
        [-e for e in entropy],
        color="tab:cyan",
        linewidth=1.5,
        label="Entropy",
    )
    (l2,) = ax2.plot(
        train_ts_k,
        std,
        color="tab:brown",
        linewidth=1.5,
        linestyle="--",
        label="Policy Std",
    )
    ax.set_title("Entropy & Policy Std")
    ax.set_xlabel("Timesteps (k)")
    ax.set_ylabel("Entropy")
    ax2.set_ylabel("Std")
    ax.legend(handles=[l1, l2], loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {args.output}")

    if ep_rew and expl_var and std:
        print(
            f"\nFinal stats: reward={ep_rew[-1]:.1f}, timesteps={timesteps[-1]}, "
            f"expl_var={expl_var[-1]:.3f}, std={std[-1]:.3f}"
        )


if __name__ == "__main__":
    main()
