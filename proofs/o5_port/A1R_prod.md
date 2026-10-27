| target | status | compile s | kernels (Pallas) | max regs/VGPR (Pallas) | STACK>0 / spill (Pallas) | max STACK | temp MB | while / cond / cmd-buf in HLO |
|---|---|---:|---:|---:|---|---:|---:|---|
| gfx942_mi300x | PASS | 226.8 | 1096 (43) | 278 | 0 (VGPR spill/scratch) | 68 | 500.596688 | 12 / 5 / 0 |
| gfx90a_mi250x | PASS | 245.0 | 1101 (43) | 278 | 0 (VGPR spill/scratch) | 132 | 624.70216 | 12 / 5 / 0 |
| gfx1100_rx7900xtx | PASS | 212.2 | 1137 (43) | 256 | 1 (VGPR spill/scratch) | 108 | 523.0292 | 12 / 5 / 0 |

- gfx1100_rx7900xtx: `b_core_coef_fp32` (acoustic) vgpr=256, vgpr_spill=26, sgpr_spill=10, scratch=108
