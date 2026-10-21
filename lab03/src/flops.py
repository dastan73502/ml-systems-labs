def linear(n_in, n_out, tokens=1):
    return 2 * tokens * n_in * n_out


def conv2d(k, c_in, c_out, h_out, w_out):
    return 2 * k * k * c_in * c_out * h_out * w_out


def attention_layer(seq, d_model, d_ff):
    return {
        "QKV projections": 2 * seq * d_model * 3 * d_model,
        "Scores QK^T": 2 * seq * seq * d_model,
        "Weighted sum AV": 2 * seq * seq * d_model,
        "Output projection": 2 * seq * d_model * d_model,
        "FFN": 2 * seq * d_model * d_ff * 2,
    }


def mlp_total():
    return {
        "Linear 784→256": linear(784, 256),
        "Linear 256→128": linear(256, 128),
        "Linear 128→10": linear(128, 10),
    }


def cnn_total():
    return {
        "Conv2d 1→16 (28×28)": conv2d(3, 1, 16, 28, 28),
        "Conv2d 16→32 (14×14)": conv2d(3, 16, 32, 14, 14),
        "Linear 1568→10": linear(32 * 7 * 7, 10),
    }


def attention_total(seq=16, d_model=64, d_ff=128, layers=2):
    parts = {"Patch embedding 49→64": linear(49, d_model, tokens=seq)}
    for i in range(1, layers + 1):
        for key, value in attention_layer(seq, d_model, d_ff).items():
            parts[f"Layer {i}: {key}"] = value
    parts["Head 64→10"] = linear(d_model, 10)
    return parts


HAND = {"MLP": mlp_total, "CNN": cnn_total, "Attention": attention_total}
