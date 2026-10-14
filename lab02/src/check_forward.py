import numpy as np

from data import RESULTS, load_data
from nn import init_params, forward, cross_entropy


def main():
    X_train, X_test, Y_train, Y_test = load_data()
    lines = [
        f"X_train {X_train.shape}  X_test {X_test.shape}",
        f"Y_train {Y_train.shape}  Y_test {Y_test.shape}",
        f"X range [{X_train.min():.2f}, {X_train.max():.2f}]",
    ]
    for width in (16, 64):
        params = init_params(width)
        lines.append(f"\nW = {width}")
        for name, value in params.items():
            lines.append(f"  {name}: {value.shape}")
        p, _ = forward(params, X_train[:32])
        row_sums_ok = np.allclose(p.sum(axis=1), 1.0, atol=1e-6)
        lines.append(f"  output shape {p.shape}, rows sum to 1: {row_sums_ok}")
        p_all, _ = forward(params, X_train)
        loss = cross_entropy(p_all, Y_train)
        lines.append(f"  untrained loss {loss:.4f}  (ln 10 = {np.log(10):.4f})")
    text = "\n".join(lines)
    print(text)
    (RESULTS / "forward_check.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
