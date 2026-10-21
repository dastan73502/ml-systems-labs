import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import RESULTS

COLORS = {"MLP": "#2a6fdb", "CNN": "#e07b39", "Attention": "#3a9e5c"}


def scatter(ax, xs, ys, names, xlabel, log_x=False):
    for x, y, name in zip(xs, ys, names):
        ax.scatter(x, y, s=120, color=COLORS[name], zorder=3)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(8, 6), fontsize=10)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Test accuracy")
    ax.grid(alpha=0.3)


def main():
    cost = json.loads((RESULTS / "cost.json").read_text())
    acc = json.loads((RESULTS / "accuracy.json").read_text())
    names = list(cost)
    test_acc = [acc[n]["test_acc"] for n in names]
    params = [cost[n]["params"] for n in names]
    lat1 = [cost[n]["latency_b1_ms"] for n in names]
    lat128 = [cost[n]["latency_b128_ms"] for n in names]
    flops = [cost[n]["hand_flops"] / 1e6 for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    scatter(axes[0], params, test_acc, names, "Parameters", log_x=True)
    axes[0].set_title("Accuracy vs parameters")
    scatter(axes[1], lat1, test_acc, names, "Latency, batch 1 (ms, median of 100)")
    axes[1].set_title("Accuracy vs latency (batch 1)")
    scatter(axes[2], lat128, test_acc, names, "Latency, batch 128 (ms, median of 20)")
    axes[2].set_title("Accuracy vs latency (batch 128)")
    fig.tight_layout()
    fig.savefig(RESULTS / "tradeoffs.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6, 4.8))
    scatter(ax, flops, test_acc, names, "Forward MFLOPs per image (hand)")
    ax.set_title("Accuracy vs compute")
    fig.tight_layout()
    fig.savefig(RESULTS / "accuracy_vs_flops.png", dpi=150)


if __name__ == "__main__":
    main()
