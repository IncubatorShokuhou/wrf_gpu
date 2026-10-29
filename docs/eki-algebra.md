# EKI observation covariance and storage

`gpuwrf.surface_params.eki.update` uses host float64 NumPy between forecast windows.
It preserves the sample-covariance EKI update and optional Tikhonov prior augmentation.
It does not import a model or add any GPU kernels, transfers, allocations or compilation.

Pass observation **variances**, not standard deviations, as `EKIProblem.obs_cov`.
The existing optimizer caller squares its `sigma` vector. Diagonal covariance stays
a vector even for full native cloud observations. For registered correlation groups:

```python
from gpuwrf.surface_params.eki import BlockDiagonalCovariance

covariance = BlockDiagonalCovariance((group1_covariance, group2_variances))
```

Blocks are contiguous in observation order. Permute predictions, truth and keys
together upstream if necessary. Every matrix block must be symmetric positive
definite and have at most 512 observations; variance-vector blocks have no such
limit. Blocks are copied and made read-only. Their total size must match the
observation vector. Correlation lengths, superobservation grouping, representative
error and cross-date correlation are scientific inputs supplied by the observation
operator. This algebra does not infer them or assume that native pixels are
independent. Superobservation errors must not shrink by an unjustified square root
of the pixel count. No localisation or change to the observation operator is made
inside the solver.

If `A` is the whitened observation anomaly matrix (members as rows, scaled by
`sqrt(dt/(J-1))`), `T` is the parameter anomaly matrix scaled by `1/sqrt(J-1)`, and
`D` is the whitened innovation including optional standard normal perturbations,
the update is `theta + (solve(I + A @ A.T, A @ D.T)).T @ T`.
TEKI appends only a parameter-sized prior block. When members outnumber augmented
observations, a thin SVD of `A` avoids a large member-sized matrix too.

Storage is `O(J*(m+p) + J*p + sum(block_size**2))`; the usual `J <= m+p` solve costs
`O((m+p)*J**2 + J**3)`. Neither the update nor its misfit diagnostic constructs a
global observation covariance or observation sample-covariance matrix. The explicit
`method="dense"` reference path and `problem.gamma` dense view refuse more than
512 observations before allocation (including the prior for the reference update).
Dense covariance inputs and individual dense correlation blocks are also bounded.

Missing/nonfinite observations, any member prediction or parameter, and empty
support are refused before reduction. The scorer must freeze a common support
upstream; this routine never silently removes, masks or imputes observations.
Zero/negative variances and singular/asymmetric covariance blocks are refused.
Rank-deficient ensemble anomalies are valid because the solve includes the identity.

Seeded perturbations now use a block Cholesky square root instead of NumPy's old
dense SVD square root. Their Gaussian distribution is the same; individual seeded
draws and resumed trajectories crossing this code change can differ. Replays within
one version are deterministic. Deterministic and identically perturbed dense checks
use `rtol=2e-11, atol=2e-12` in float64 across both aspect ratios, rank-deficient
ensembles, diagonal/dense/block errors, tempering and TEKI.

Reproduce CPU-only algebra scaling with `proofs/v45_ekialg/scaling.py`, once per
fresh process and matrix shape. It reports median update wall time and Linux peak
RSS including input construction and imports. Synthetic algebra measurements are
not forecast timings or scientific cloud-skill evidence.

## Measured CPU algebra

Source `0a067d3a22`; fresh process per row, three updates, float64, four parameters,
TEKI and perturbed observations ON, `dt=0.2`. Single-thread BLAS, shared CPU13/29,
nice19, MemoryMax8G/Swap0. RSS includes imports and input construction. These are
shared-host measurements; no forecast speed or contention-free timing claim.

| Observations | Members | Error covariance | Median update (ms) | Peak RSS (MiB) |
|---:|---:|---|---:|---:|
| 512 | 64 | diagonal | 0.900 | 158.8 |
| 4,096 | 64 | diagonal | 7.474 | 170.9 |
| 25,200 | 16 | diagonal | 10.800 | 179.6 |
| 25,200 | 32 | diagonal | 18.748 | 201.3 |
| 25,200 | 64 | diagonal | 35.354 | 244.4 |
| 100,800 | 64 | diagonal | 1236.237 | 512.6 |
| 25,200 | 64 | 8 by 8 blocks | 161.240 | 253.0 |
| 100,800 | 64 | 8 by 8 blocks | 1005.314 | 500.2 |

The 25,200-observation, 64-member gate passes the 2,000,000,000-byte RSS cap
for both covariance structures. One old dense float64 matrix at that observation
count alone would require 5,080,320,000 bytes [calculated]. At 100,800 observations,
peak RSS remains below 538 MB. Timing at that size varied substantially
(0.95–1.71 s across the diagonal/block samples); do not extrapolate throughput.
Raw measurements: `proofs/v45_ekialg/receipts/SCALING.json`.
