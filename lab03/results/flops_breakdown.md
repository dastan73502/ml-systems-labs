### MLP

| Layer | FLOPs |
|---|---|
| Linear 784→256 | 401,408 |
| Linear 256→128 | 65,536 |
| Linear 128→10 | 2,560 |
| **Total** | **469,504** |

### CNN

| Layer | FLOPs |
|---|---|
| Conv2d 1→16 (28×28) | 225,792 |
| Conv2d 16→32 (14×14) | 1,806,336 |
| Linear 1568→10 | 31,360 |
| **Total** | **2,063,488** |

### Attention

| Layer | FLOPs |
|---|---|
| Patch embedding 49→64 | 100,352 |
| Layer 1: QKV projections | 393,216 |
| Layer 1: Scores QK^T | 32,768 |
| Layer 1: Weighted sum AV | 32,768 |
| Layer 1: Output projection | 131,072 |
| Layer 1: FFN | 524,288 |
| Layer 2: QKV projections | 393,216 |
| Layer 2: Scores QK^T | 32,768 |
| Layer 2: Weighted sum AV | 32,768 |
| Layer 2: Output projection | 131,072 |
| Layer 2: FFN | 524,288 |
| Head 64→10 | 1,280 |
| **Total** | **2,329,856** |

