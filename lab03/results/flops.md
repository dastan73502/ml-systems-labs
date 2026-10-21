| Model | Params (torchinfo) | Hand FLOPs | thop MACs | thop ×2 | thop ×2 / hand | PyTorch FlopCounter | FlopCounter / hand |
|---|---|---|---|---|---|---|---|
| MLP | 235,146 | 469,504 | 234,752 | 469,504 | 1.000 | 469,504 | 1.000 |
| CNN | 20,490 | 2,063,488 | 1,031,744 | 2,063,488 | 1.000 | 2,063,488 | 1.000 |
| Attention | 71,818 | 2,329,856 | 591,488 | 1,182,976 | 0.508 | 2,198,784 | 0.944 |
