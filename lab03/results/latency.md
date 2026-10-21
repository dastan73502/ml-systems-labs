torch 2.14.1+cpu, threads 8, AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD, Python 3.11.8

| Model | Batch 1 median (ms) | Batch 1 IQR (ms) | Batch 128 median (ms) | Batch 128 IQR (ms) | Per-image @128 (ms) | Throughput @128 (img/s) |
|---|---|---|---|---|---|---|
| MLP | 0.0951 | 0.0943–0.0957 | 0.254 | 0.250–0.257 | 0.0020 | 504,632 |
| CNN | 0.2230 | 0.2203–0.2249 | 7.136 | 6.710–7.338 | 0.0558 | 17,936 |
| Attention | 0.6762 | 0.6685–0.6813 | 3.423 | 2.857–3.600 | 0.0267 | 37,390 |
