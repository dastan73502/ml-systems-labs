| Type | Layer | Formula | FLOPs |
|---|---|---|---|
| Linear | MLP Linear 784→256 | `2 × 784 × 256` | 401,408 |
| Conv2d | CNN Conv2d 16→32, 3×3, out 14×14 | `2 × 3 × 3 × 16 × 32 × 14 × 14` | 1,806,336 |
| Attention | QKV projections | `2 × 16 × 64 × 3 × 64` | 393,216 |
| Attention | Scores QK^T | `2 × 16² × 64` | 32,768 |
| Attention | Weighted sum AV | `2 × 16² × 64` | 32,768 |
| Attention | Output projection | `2 × 16 × 64 × 64` | 131,072 |
| Attention | Self-attention block total | `sum of four rows above` | 589,824 |
| Attention | FFN 64→128→64 | `2 × 16 × 64 × 128 × 2` | 524,288 |
| Attention | Encoder layer total | `attention + FFN` | 1,114,112 |
