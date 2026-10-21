import json
import time

import torch
from torch import nn

from data import RESULTS, load_loaders
from models import MODELS, build

EPOCHS = 5
LR = 1e-3


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            loss_sum += criterion(logits, y).item()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.numel()
    return loss_sum / total, correct / total


def train(name, loaders, log):
    train_loader, val_loader, test_loader = loaders
    model = build(name)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    history = []
    start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.numel()
            seen += y.numel()
        val_loss, val_acc = evaluate(model, val_loader)
        history.append({"epoch": epoch, "train_loss": loss_sum / seen, "val_loss": val_loss, "val_acc": val_acc})
        log(f"{name:<9} epoch {epoch}/{EPOCHS}  train loss {loss_sum / seen:.4f}  val loss {val_loss:.4f}  val acc {val_acc:.4f}")
    train_time = time.perf_counter() - start
    _, test_acc = evaluate(model, test_loader)
    log(f"{name:<9} test acc {test_acc:.4f}  train time {train_time:.1f} s")
    return {"history": history, "test_acc": test_acc, "train_time_s": train_time}


def main():
    torch.manual_seed(0)
    loaders = load_loaders()
    log_lines = []

    def log(line):
        print(line)
        log_lines.append(line)

    results = {name: train(name, loaders, log) for name in MODELS}
    table = ["| Model | Val acc (epoch 5) | Test acc | Train time 5 epochs (s) |", "|---|---|---|---|"]
    for name, r in results.items():
        table.append(f"| {name} | {r['history'][-1]['val_acc']:.4f} | {r['test_acc']:.4f} | {r['train_time_s']:.1f} |")
    (RESULTS / "accuracy.md").write_text("\n".join(table) + "\n")
    (RESULTS / "train_log.txt").write_text("\n".join(log_lines) + "\n")
    (RESULTS / "accuracy.json").write_text(json.dumps(results, indent=2))
    print("\n".join(table))


if __name__ == "__main__":
    main()
