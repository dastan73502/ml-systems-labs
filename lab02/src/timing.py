import time

import numpy as np

from data import RESULTS, load_data
from nn import init_params, forward, backward


def median_time(fn, runs=100, warmup=10):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def matmul_flops(batch, width, n_in=64, n_out=10):
    forward_flops = 2 * batch * (n_in * width + width * width + width * n_out)
    backward_flops = 2 * batch * (
        width * n_out
        + width * n_out
        + width * width
        + width * width
        + n_in * width
    )
    return forward_flops, backward_flops


def forward_matmuls(params, X, a1, a2):
    X @ params["W1"]
    a1 @ params["W2"]
    a2 @ params["W3"]


def backward_matmuls(params, cache, d3, d2, d1):
    cache["a2"].T @ d3
    d3 @ params["W3"].T
    cache["a1"].T @ d2
    d2 @ params["W2"].T
    cache["X"].T @ d1


def main():
    X_train, _, Y_train, _ = load_data()
    rng = np.random.default_rng(0)
    big = rng.choice(len(X_train), 8192)
    configs = [(32, X_train[:32], Y_train[:32]), (1257, X_train, Y_train), (8192, X_train[big], Y_train[big])]
    lines = [
        "| Width | Batch | Forward (ms) | Backward (ms) | Bwd/Fwd | Fwd matmul (ms) | Bwd matmul (ms) | Fwd GFLOP | Bwd GFLOP |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for width in (16, 64):
        params = init_params(width)
        for batch_size, X, Y in configs:
            _, cache = forward(params, X)
            d3 = (cache["p"] - Y) / batch_size
            d2 = (d3 @ params["W3"].T) * (cache["z2"] > 0)
            d1 = (d2 @ params["W2"].T) * (cache["z1"] > 0)
            t_fwd = median_time(lambda: forward(params, X))
            t_bwd = median_time(lambda: backward(params, cache, Y))
            t_fwd_mm = median_time(lambda: forward_matmuls(params, X, cache["a1"], cache["a2"]))
            t_bwd_mm = median_time(lambda: backward_matmuls(params, cache, d3, d2, d1))
            f_flops, b_flops = matmul_flops(batch_size, width)
            lines.append(
                f"| {width} | {batch_size} | {t_fwd * 1e3:.4f} | {t_bwd * 1e3:.4f} | {t_bwd / t_fwd:.2f} "
                f"| {t_fwd_mm * 1e3:.4f} | {t_bwd_mm * 1e3:.4f} | {f_flops / 1e9:.4f} | {b_flops / 1e9:.4f} |"
            )
    table = "\n".join(lines)
    (RESULTS / "timing.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
