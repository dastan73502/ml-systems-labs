import numpy as np

from data import RESULTS, load_data
from nn import init_params, forward, backward


def per_sample_loss(params, X, Y):
    a1 = np.maximum(0, X @ params["W1"] + params["b1"])
    a2 = np.maximum(0, a1 @ params["W2"] + params["b2"])
    z3 = a2 @ params["W3"] + params["b3"]
    z3 = z3 - z3.max(axis=1, keepdims=True)
    log_p = z3 - np.log(np.exp(z3).sum(axis=1, keepdims=True))
    return -np.sum(Y * log_p, axis=1)


def grad_check(params, X, Y, n_checks=10, eps=1e-5, seed=1):
    rng = np.random.default_rng(seed)
    _, cache = forward(params, X)
    grads = backward(params, cache, Y)
    weight_names = ["W1", "W2", "W3"]
    results = []
    for _ in range(n_checks):
        name = weight_names[rng.integers(len(weight_names))]
        idx = tuple(int(rng.integers(s)) for s in params[name].shape)
        original = params[name][idx]
        params[name][idx] = original + eps
        loss_plus = per_sample_loss(params, X, Y)
        params[name][idx] = original - eps
        loss_minus = per_sample_loss(params, X, Y)
        params[name][idx] = original
        numeric = np.mean(loss_plus - loss_minus) / (2 * eps)
        analytic = grads[name][idx]
        denom = max(abs(numeric), abs(analytic), 1e-12)
        rel_err = abs(numeric - analytic) / denom
        results.append((name, idx, analytic, numeric, rel_err))
    return results


def main():
    X_train, _, Y_train, _ = load_data()
    all_passed = True
    lines = []
    for width in (16, 64):
        params = init_params(width)
        results = grad_check(params, X_train, Y_train)
        passed = sum(r[4] < 1e-6 for r in results)
        lines.append(f"\nW = {width}")
        lines.append(f"{'param':<6}{'index':<12}{'analytic':>16}{'numerical':>16}{'rel error':>12}")
        for name, idx, a, n, e in results:
            lines.append(f"{name:<6}{str(idx):<12}{a:>16.8e}{n:>16.8e}{e:>12.2e}")
        status = "PASS" if passed >= 9 else "FAIL"
        lines.append(f"{passed}/10 weights with rel error < 1e-6 -> {status}")
        all_passed &= passed >= 9
    lines.append(f"\nGradient check overall: {'PASS' if all_passed else 'FAIL'}")
    text = "\n".join(lines)
    print(text)
    (RESULTS / "grad_check.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
