import json
import platform
import time

import numpy as np
import torch
from thop import profile
from torch.utils.flop_counter import FlopCounterMode
from torchinfo import summary

from data import RESULTS
from flops import HAND, attention_layer, conv2d, linear
from models import MODELS, build


def torchinfo_params(model):
    stats = summary(model, input_size=(1, 1, 28, 28), verbose=0)
    return stats.total_params, stats


def thop_flops(model):
    macs, _ = profile(model, inputs=(torch.randn(1, 1, 28, 28),), verbose=False)
    return 2 * macs, macs


def pytorch_flops(model):
    counter = FlopCounterMode(display=False)
    with counter:
        model(torch.randn(1, 1, 28, 28))
    return counter.get_total_flops()


def median_latency(model, batch_size, runs, warmup=10):
    x = torch.randn(batch_size, 1, 28, 28)
    times = []
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        for _ in range(runs):
            start = time.perf_counter()
            model(x)
            times.append(time.perf_counter() - start)
    return float(np.median(times)), float(np.percentile(times, 25)), float(np.percentile(times, 75))


def single_layer_table():
    rows = [
        ("Linear", "MLP Linear 784→256", "2 × 784 × 256", linear(784, 256)),
        ("Conv2d", "CNN Conv2d 16→32, 3×3, out 14×14", "2 × 3 × 3 × 16 × 32 × 14 × 14", conv2d(3, 16, 32, 14, 14)),
    ]
    att = attention_layer(16, 64, 128)
    rows += [
        ("Attention", "QKV projections", "2 × 16 × 64 × 3 × 64", att["QKV projections"]),
        ("Attention", "Scores QK^T", "2 × 16² × 64", att["Scores QK^T"]),
        ("Attention", "Weighted sum AV", "2 × 16² × 64", att["Weighted sum AV"]),
        ("Attention", "Output projection", "2 × 16 × 64 × 64", att["Output projection"]),
        ("Attention", "Self-attention block total", "sum of four rows above", sum(v for k, v in att.items() if k != "FFN")),
        ("Attention", "FFN 64→128→64", "2 × 16 × 64 × 128 × 2", att["FFN"]),
        ("Attention", "Encoder layer total", "attention + FFN", sum(att.values())),
    ]
    lines = ["| Type | Layer | Formula | FLOPs |", "|---|---|---|---|"]
    for kind, name, formula, value in rows:
        lines.append(f"| {kind} | {name} | `{formula}` | {value:,} |")
    return "\n".join(lines)


def main():
    torch.set_num_threads(torch.get_num_threads())
    results = {}
    flop_lines = [
        "| Model | Params (torchinfo) | Hand FLOPs | thop MACs | thop ×2 | thop ×2 / hand | PyTorch FlopCounter | FlopCounter / hand |",
        "|---|---|---|---|---|---|---|---|",
    ]
    latency_lines = [
        "| Model | Batch 1 median (ms) | Batch 1 IQR (ms) | Batch 128 median (ms) | Batch 128 IQR (ms) | Per-image @128 (ms) | Throughput @128 (img/s) |",
        "|---|---|---|---|---|---|---|",
    ]
    summaries = []
    breakdown = []
    for name in MODELS:
        model = build(name).eval()
        params, stats = torchinfo_params(model)
        summaries.append(f"## {name}\n\n```\n{stats}\n```\n")
        hand_parts = HAND[name]()
        hand = sum(hand_parts.values())
        breakdown.append(f"### {name}\n\n| Layer | FLOPs |\n|---|---|")
        breakdown += [f"| {k} | {v:,} |" for k, v in hand_parts.items()]
        breakdown.append(f"| **Total** | **{hand:,}** |\n")
        thop2, macs = thop_flops(build(name).eval())
        counted = pytorch_flops(model)
        b1, b1_q1, b1_q3 = median_latency(model, 1, runs=100)
        b128, b128_q1, b128_q3 = median_latency(model, 128, runs=20)
        results[name] = {
            "params": params,
            "hand_flops": hand,
            "thop_macs": int(macs),
            "thop_flops": int(thop2),
            "flopcounter_flops": int(counted),
            "latency_b1_ms": b1 * 1e3,
            "latency_b128_ms": b128 * 1e3,
        }
        flop_lines.append(
            f"| {name} | {params:,} | {hand:,} | {int(macs):,} | {int(thop2):,} | {thop2 / hand:.3f} "
            f"| {int(counted):,} | {counted / hand:.3f} |"
        )
        latency_lines.append(
            f"| {name} | {b1 * 1e3:.4f} | {b1_q1 * 1e3:.4f}–{b1_q3 * 1e3:.4f} | {b128 * 1e3:.3f} "
            f"| {b128_q1 * 1e3:.3f}–{b128_q3 * 1e3:.3f} | {b128 * 1e3 / 128:.4f} | {128 / b128:,.0f} |"
        )
    env = (
        f"torch {torch.__version__}, threads {torch.get_num_threads()}, "
        f"{platform.processor() or platform.machine()}, Python {platform.python_version()}"
    )
    (RESULTS / "single_layer_flops.md").write_text(single_layer_table() + "\n")
    (RESULTS / "flops_breakdown.md").write_text("\n".join(breakdown) + "\n")
    (RESULTS / "flops.md").write_text("\n".join(flop_lines) + "\n")
    (RESULTS / "latency.md").write_text(env + "\n\n" + "\n".join(latency_lines) + "\n")
    (RESULTS / "torchinfo_summary.md").write_text("\n".join(summaries))
    (RESULTS / "cost.json").write_text(json.dumps(results, indent=2))
    print(single_layer_table())
    print()
    print("\n".join(flop_lines))
    print()
    print(env)
    print("\n".join(latency_lines))


if __name__ == "__main__":
    main()
