import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import RESULTS, load_data
from nn import init_params, forward, backward, cross_entropy, accuracy


def train(width, lr, X_train, Y_train, X_test, Y_test, epochs=50, batch_size=32, std=0.01, seed=0, log=print):
    params = init_params(width, std=std, seed=seed)
    rng = np.random.default_rng(seed)
    n = X_train.shape[0]
    history = []
    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch = order[start:start + batch_size]
            _, cache = forward(params, X_train[batch])
            grads = backward(params, cache, Y_train[batch])
            for key in params:
                params[key] -= lr * grads[key]
        p_train, _ = forward(params, X_train)
        train_loss = cross_entropy(p_train, Y_train)
        history.append(train_loss)
        log(f"std={std} W={width:<3} lr={lr:<6} epoch {epoch:>2}/{epochs}  train loss {train_loss:.4f}")
    test_acc = accuracy(params, X_test, Y_test)
    return history, test_acc


def run_experiment(std, plot_name, table_name, data, log, learning_rates=(0.001, 0.01, 0.1), widths=(16, 64)):
    X_train, X_test, Y_train, Y_test = data
    runs = {}
    for width in widths:
        for lr in learning_rates:
            runs[(width, lr)] = train(width, lr, X_train, Y_train, X_test, Y_test, std=std, log=log)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, width in zip(axes, widths):
        for lr in learning_rates:
            history, test_acc = runs[(width, lr)]
            ax.plot(range(1, len(history) + 1), history, label=f"lr={lr} (test acc {test_acc:.3f})")
        ax.set_title(f"Hidden width W = {width}, init std = {std}")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("Training cross-entropy loss")
    fig.tight_layout()
    fig.savefig(RESULTS / plot_name, dpi=150)
    plt.close(fig)

    lines = ["| Width | Learning rate | Final train loss | Test accuracy |", "|---|---|---|---|"]
    for width in widths:
        for lr in learning_rates:
            history, test_acc = runs[(width, lr)]
            lines.append(f"| {width} | {lr} | {history[-1]:.4f} | {test_acc:.4f} |")
    table = "\n".join(lines)
    (RESULTS / table_name).write_text(table + "\n")
    log(f"\ninit std = {std}\n{table}\n")


def main():
    data = load_data()
    with open(RESULTS / "train_log.txt", "w") as f:
        def log(line):
            print(line)
            f.write(line + "\n")

        run_experiment(0.01, "loss_curves.png", "accuracy.md", data, log)
        run_experiment(0.1, "loss_curves_std0.1.png", "accuracy_std0.1.md", data, log)


if __name__ == "__main__":
    main()
