| target | status | compile s | kernels (Pallas) | max regs/VGPR (Pallas) | STACK>0 / spill (Pallas) | max STACK | temp MB | while / cond / cmd-buf in HLO |
|---|---|---:|---:|---:|---|---:|---:|---|
| sm80_a100 | PASS | 281.6 | 1218 (43) | 206 | 1 | 32 | 881.359312 | 12 / 5 / 0 |
| sm120_rtx5090_captured | PASS | 282.4 | 1103 (43) | 200 | 1 | 32 | 523.778768 | 12 / 5 / 0 |
| sm86_rtx3090 | PASS | 282.8 | 1205 (43) | 210 | 0 | 32 | 902.771152 | 12 / 5 / 0 |
| sm89_rtx4090 | PASS | 253.9 | 1155 (43) | 210 | 0 | 32 | 500.351696 | 12 / 5 / 0 |
| sm90_h100 | PASS | 222.2 | 1131 (43) | 206 | 1 | 32 | 548.979408 | 12 / 5 / 0 |
| sm75_t4 | **FAIL** | – | – | – | – | – | – | JaxRuntimeError: FAILED_PRECONDITION: Triton support is only enabled for Ampere GPUs (compute capability 8.0) and up, bu |

- sm80_a100: `b_core_w_fp32` (acoustic) regs=72, stack=8, local=0
- sm120_rtx5090_captured: `kf_column_fp32` (KF) regs=168, stack=8, local=0
- sm90_h100: `kf_column_fp32` (KF) regs=168, stack=8, local=0
