# o2-multigpu evidence (CPU virtual ranks; no GPU used)

The table below preserves the original author's historical results. Its comparator used numeric equality;
the follow-up o2-mgpufix comparator requires literal bytes, equal schema and finite values on BOTH arms.
The critic independently confirmed literal BYTE for the original 1×1/2×1/2×2 three-step states. Fresh
o2-mgpufix receipts live under `<USER_HOME>/wrf_gpu2_lanes/o2-mgpufix/GF04-real/`; never interpret matching
NaNs, omitted leaves or signed-zero differences as BITWISE. These comparisons are partition/regression
evidence against the plain port loop, not fidelity evidence against pristine CPU-WRF.

The CLI uses the selected-configuration cone, including topographic shadow followed by dynamics, and
refuses undersized halos, global-reduction paths and census-enabled decomposition. Historical H20 arms
below are CPU lab measurements; they do not qualify release GPU kernels or the F2 gate.

Tools: `virtual_rank_check.py` (real case, single-device reference + automatic 1×1 same-program control + arms),
`noise_pattern.py` (where round-off differences sit), `release_bytes_probe.py` (halo bytes/column of the release
carry), `inventory_probe.py` (carry/namelist leaf inventory). Artifacts: `<USER_HOME>/wrf_gpu2_lanes/o2-multigpu/vr/`.
Case: PROD d01 (120×70×44, dt 54 s, Thompson/MYNN/Noah-MP/RRTMG/KF, wrfinput+wrfbdy), legacy XLA path
(`GPUWRF_FAST_DEFAULTS=0`: the release Triton kernels need a GPU or Pallas interpret mode).
Reference = plain `operational_mode._advance_chunk_fori` on one device; "bitwise" = every carry leaf identical.

| arm | steps | result vs plain single-device run [M] |
|---|---|---|
| 1×1 tile (tile == domain), traced trip count | 1 / 3 (incl. radiation) | BITWISE 167/167 |
| 1×1 tile, CONSTANT trip count (first version) | 1 | 140/167 (XLA:CPU round-off) → fixed by tracing n_steps |
| 2×1, halo 5 / 10 | 1 | localized halo errors at the tile boundary (near/far 4e12 / 4.5e8) — short-halo mutant FAILS |
| 2×1, halo 15 | 1 | round-off only (≤1e-13 rel) |
| 2×1, halo 20 / 25 | 1 | BITWISE 167/167 |
| 2×1, halo 38 (analytic bound) | 1 | 147/167: ~1e-15 rel only in the last 8 columns of the true east edge (tile-width codegen; not near tile boundaries) |
| 2×1, halo 20 | 3 (radiation + 2 exchanges consumed) | BITWISE 167/167 |
| 2×2 (P=4, corner messages), halo 20 | 3 | BITWISE 167/167 |
| 2×1 live-leaf exchange (127/167 leaves; 18.4 vs 24.2 MB per message) | 3 | BITWISE 167/167 |
| MUTANT 2×2 without diagonal (corner) messages | 3 | FAILS 140/167, differences only near the tile boundaries |
| MUTANT 2×1 true-edge boundary strips sliced one cell off | 3 | FAILS 82/167, 1.06 M non-finite values |
| RunPod bench script end to end (`decomposed_bench.py --allow-cpu`), Swiss mirror 2×2 = 84×84, 2×1, halo 20, ppermute transport, live leaves | 3 | Tier-P: 106/154 bitwise, rest ≤6.5e-12 rel, all in x 64–83 (tile 1's true east edge), none near the internal cut 42±20 |
| 4×2 (P=8), halo 20 | 3 | Strict MISMATCH 117/167; finite. Fresh fix-round replay reproduces every original mismatch metric and the 64-bit raw-state hash prefix. Other fp64 rows ≤1.478e-11 relative; h_diabatic differs by 3.158e-15 K/s (relative1.5 near zero), and packed float32 energy accumulators by0.0009765625 (relative1.314e-8). No blanket relative bound or automatic Tier-P pass. |

Toy-operator tests (`tests/test_parallel_decomposition.py`, 2–6 virtual devices, both transports, k = 1/2):
exact from the derived minimal halo, wrong one cell below; live-leaf exchange exact with fewer bytes; CLI hook
scatter/advance/gather exact.

Halo errors look different: at halo 10 (one step) the boundary differences are 1e-4…0.4 absolute with a
near/far ratio of 1e8. The fresh H5 three-step negative is finite and rejects107/167: max u/v differences
0.707/0.711m/s and theta0.558K. The old blanket P8 near/far and relative summaries omit the near-zero and
float32 exceptions above; complete metrics remain in the fix-round receipt. Not run: a 4×1 / halo-25 control that
would attribute the P=8 round-off to tile width directly, the release (Pallas) kernels on CPU, and the unmodified 42×42 Swiss
case (smaller than one tile plus halo; it is run mirror-tiled above and in RunPod T3a).

Tile-width round-off pattern (cause not isolated): bitwise for tile widths 80 and 85 (PROD), differing at ~1e-15 per
step for widths 62/70/75/98, always in the tile that holds the true EAST edge, near that edge. Elementwise
transcendentals and simple fused arithmetic are bitwise at all these widths (probes in the lane log).
The fix round additionally passed the actual Swiss42×42 case at2×1/H38:154/154 literal BYTE,
finite, with identical timed steps1–3 and radiation cadence33 in both benchmark arms. The frozen release
jaxpr identities remain Swiss2bf12cb8 / PROD6fc715a2. Source/cache/acceptance fixes passed60 targeted
checks; removing the comparator, shadow bound or caller alias protection yields9/1/1 genuine assertions.

Author raw references were reused through the fresh1×1 byte control, then removed at the manager's request
after the last reader. Small fix-round receipts live under `<USER_HOME>/wrf_gpu2_lanes/o2-mgpufix/`;
the original P8 JSON is retained as `GF04-P8-PRIOR-REPORT.json` with a SHA provenance record. All comparisons
remain CPU partition regression evidence. The strict P8 mismatch is preserved and GPU/F2 qualification is pending.
