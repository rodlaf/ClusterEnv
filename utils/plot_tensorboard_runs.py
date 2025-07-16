import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# === Configurable KL values for filtering/legend ===
KL_THRESHOLDS = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0]


def extract_scalar_from_event(dir_path, tag):
    ea = EventAccumulator(dir_path)
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return [], []

    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values


def extract_kl_threshold(run_dir_name):
    match = re.search(r"kl[\-_]?([0-9.]+)", run_dir_name, re.IGNORECASE)
    if match:
        return float(match.group(1))
    else:
        return None


def smooth(values, window):
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window)/window, mode='valid')


def rolling_median(values, window):
    if len(values) < window:
        return values
    return [np.median(values[i:i+window]) for i in range(len(values)-window+1)]


def plot_metric_across_runs(root_dir, tag, ylabel, filename, smooth_median=False, median_window=200):
    plt.figure(figsize=(10, 6))

    for subdir in sorted(os.listdir(root_dir)):
        run_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(run_path):
            continue

        kl = extract_kl_threshold(subdir)
        if kl not in KL_THRESHOLDS:
            continue

        try:
            steps, values = extract_scalar_from_event(run_path, tag)
            if not steps or not values:
                continue

            if smooth_median and tag == "charts/episodic_return":
                values = rolling_median(values, median_window)
                steps = steps[:len(values)]

            plt.plot(steps, values, label=f"KL = {kl}")
        except Exception as e:
            print(f"[Warning] Failed to load from {subdir}: {e}")

    plt.xlabel("Global Steps", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(ylabel, fontsize=14)
    plt.legend(title="KL Threshold", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(root_dir, filename), dpi=300)
    print(f"[Saved] {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("logdir", type=str, help="Path to parent TensorBoard logs directory")
    parser.add_argument("--median_window", type=int, default=200, help="Rolling median window size for smoothing")
    args = parser.parse_args()

    plot_metric_across_runs(args.logdir, "charts/episodic_return", "Episodic Return", "reward_curve_lunarlander.png", smooth_median=True, median_window=args.median_window)
    plot_metric_across_runs(args.logdir, "charts/weight_syncs", "Weight Sync Count (Per Worker)", "sync_counts.png")
    plot_metric_across_runs(args.logdir, "charts/SPS", "Steps Per Second (SPS)", "sps_plot.png")
