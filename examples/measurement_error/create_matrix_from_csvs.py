import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tyro
import re
from dataclasses import dataclass


@dataclass
class Args:
    """Script arguments"""

    input_dir: str = "gauntlet_results_csv"
    """Directory containing the indexed CSV files from the gauntlet runs."""
    output_dir: str = "gauntlet_matrix_results"
    """Directory to save the final matrix CSV and heatmap plot."""


def parse_indexed_filename(filename: str):
    """
    Parses 'Prot_I_in_Env_J.csv' to get protagonist and environment indices.
    """
    base = os.path.basename(filename)
    match = re.match(r"Prot_(\d+)_in_Env_(\d+)\.csv", base)
    if not match:
        return None, None

    prot_idx = int(match.group(1))
    env_idx = int(match.group(2))
    return prot_idx, env_idx


def calculate_power_gain(filepath: str) -> float:
    """
    Loads a CSV and computes the mean power gain vs. baseline for the episode.
    """
    try:
        df = pd.read_csv(filepath)
        if "power_agent" not in df.columns or "power_baseline" not in df.columns:
            return np.nan

        total_agent_power = df["power_agent"].mean()
        total_baseline_power = df["power_baseline"].mean()

        if pd.isna(total_baseline_power) or total_baseline_power == 0:
            return np.nan

        return (total_agent_power / total_baseline_power) - 1
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return np.nan


def main(args: Args):
    """
    Main function to find evaluation results, build a matrix, and plot a heatmap.
    """
    os.makedirs(args.output_dir, exist_ok=True)

    csv_files = glob.glob(os.path.join(args.input_dir, "Prot_*.csv"))
    if not csv_files:
        print(
            f"No correctly formatted CSV files (e.g., 'Prot_0_in_Env_0.csv') found in '{args.input_dir}'"
        )
        return

    # {(prot_idx, env_idx): gain}
    results = {}
    max_prot_idx = -1
    max_env_idx = -1

    for filepath in csv_files:
        prot_idx, env_idx = parse_indexed_filename(filepath)
        if prot_idx is None:
            continue

        gain = calculate_power_gain(filepath)
        results[(prot_idx, env_idx)] = gain

        if prot_idx > max_prot_idx:
            max_prot_idx = prot_idx
        if env_idx > max_env_idx:
            max_env_idx = env_idx

    # --- Create the results matrix with chronological ordering ---
    num_prots = max_prot_idx + 1
    num_envs = max_env_idx + 1

    if num_prots == 0 or num_envs == 0:
        print("No valid results found to create a matrix.")
        return

    num_adversaries = num_envs - 2  # Subtract 'Clean' and 'Procedural'

    # Generate labels based on indices, ensuring correct chronological order
    prot_labels = [f"Protagonist {i}" for i in range(num_prots)]
    env_labels = ["Clean", "Procedural"]
    if num_adversaries > 0:
        env_labels += [f"Adversary {i}" for i in range(num_adversaries)]

    matrix_df = pd.DataFrame(index=prot_labels, columns=env_labels, dtype=str)
    numeric_matrix = pd.DataFrame(index=prot_labels, columns=env_labels, dtype=float)

    for (prot_idx, env_idx), gain in results.items():
        if prot_idx < num_prots and env_idx < num_envs:
            prot_label = prot_labels[prot_idx]
            env_label = env_labels[env_idx]

            # Use a placeholder for NaN values for better display
            if np.isnan(gain):
                matrix_df.loc[prot_label, env_label] = "N/A"
            else:
                matrix_df.loc[prot_label, env_label] = f"{gain:.3f}"
            numeric_matrix.loc[prot_label, env_label] = gain

    matrix_csv_path = os.path.join(args.output_dir, "power_gain_matrix.csv")
    matrix_df.to_csv(matrix_csv_path)

    print("\n" + "=" * 25 + " GAUNTLET RESULTS (Power Gain vs Baseline) " + "=" * 25)
    print(matrix_df)
    print("=" * 82)

    # --- Create and save the heatmap ---
    fig_width = max(10, num_envs)
    fig_height = max(7, num_prots * 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    sns.heatmap(
        numeric_matrix,
        annot=matrix_df.values,
        fmt="s",
        cmap="viridis",
        ax=ax,
        cbar=True,
        linewidths=0.5,
        cbar_kws={"label": "Mean Episodic Power Gain vs Baseline"},
        annot_kws={"size": 12},
    )
    ax.set_title(
        "Agent Robustness Evaluation Matrix", fontsize=16, fontweight="bold", pad=20
    )
    ax.set_xlabel("Evaluation Environment", fontsize=12)
    ax.set_ylabel("Protagonist Agent", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    plot_path = os.path.join(args.output_dir, "power_gain_heatmap.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\n✅ Analysis complete. Matrix and heatmap saved to '{args.output_dir}'")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
