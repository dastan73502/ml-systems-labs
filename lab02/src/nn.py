import numpy as np


def relu(z):
    return np.maximum(0, z)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def init_params(width, n_in=64, n_out=10, std=0.01, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0, std, (n_in, width)),
        "b1": np.zeros(width),
        "W2": rng.normal(0, std, (width, width)),
        "b2": np.zeros(width),
        "W3": rng.normal(0, std, (width, n_out)),
        "b3": np.zeros(n_out),
    }


def forward(params, X):
    z1 = X @ params["W1"] + params["b1"]
    a1 = relu(z1)
    z2 = a1 @ params["W2"] + params["b2"]
    a2 = relu(z2)
    z3 = a2 @ params["W3"] + params["b3"]
    p = softmax(z3)
    cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "p": p}
    return p, cache


def cross_entropy(p, Y, eps=1e-12):
    return -np.mean(np.sum(Y * np.log(p + eps), axis=1))


def backward(params, cache, Y):
    n = Y.shape[0]
    d3 = (cache["p"] - Y) / n
    grads = {"W3": cache["a2"].T @ d3, "b3": d3.sum(axis=0)}
    d2 = (d3 @ params["W3"].T) * (cache["z2"] > 0)
    grads["W2"] = cache["a1"].T @ d2
    grads["b2"] = d2.sum(axis=0)
    d1 = (d2 @ params["W2"].T) * (cache["z1"] > 0)
    grads["W1"] = cache["X"].T @ d1
    grads["b1"] = d1.sum(axis=0)
    return grads


def loss_fn(params, X, Y):
    p, _ = forward(params, X)
    return cross_entropy(p, Y)


def accuracy(params, X, Y):
    p, _ = forward(params, X)
    return np.mean(p.argmax(axis=1) == Y.argmax(axis=1))
