# P3 reference disposition

Scope: original-WRF reference repair and retained-artifact qualification. No production model or test file change is proposed.

The upstream MYNN cold-start RMOL read is located at `phys/MYNN-EDMF/module_bl_mynnedmf.F90:1842` (declaration473, coldcall719, firstassignment879). The port already uses zero there. See `UPSTREAM_WRF_BUGS.md`.

All four compiler-ZERO original references use admitted executable `9a67a93d...`, unchanged pristine bodies, FP traps off, and the exact sealed namelist/IC/BC bytes. Each has 25 finite hourly frames. The inactive MP40 compiler-flag exception is recorded. `reference_control/ZERO_REFERENCE_INDEX.json` binds the 100 preserved frames and 36,875 typed field-frames.

| Case | MP | Retained GPU vs ZERO | Raw-original vs ZERO RAINNC RMSE h24 (mm) | Raw-original vs ZERO T2 RMSE h24 (K) |
|---|---:|---|---:|---:|
| c21 | 50 | 25 frames, 0 bounded misses; INCOMPLETE:6 missing fields | 4.566383 | 3.134493 |
| c22 | 51 | 25 frames, 0 bounded misses; INCOMPLETE:7 missing fields | 14.283764 | 5.504015 |
| c23 | 52 | Initial frame only, 0 bounded misses; INCOMPLETE | 1.995055 | 2.298286 |
| c24 | 53 | 0 frames; INCOMPLETE, no numerical score | 8.148136 | 2.013006 |

All values above are measured. Frozen scientific manifest SHA is `9df4fab8514523286f07b127eb8bf57a7545459c4448fef721ac44c0ecc351dc`. c21/c22 had 92/142 bounded misses against raw initialization; the full original-vs-ZERO CPU sensitivity reproduces those 92/142 misses. c23/c24 sensitivity has 82/118 misses. This empirical initialization spread is disclosed, without treating it as a statistical forecast-error bound.

The retained c21 native GPU cubin replay saved 45 finite outputs from three calls. It has 240 strict column violations; a same-input 44-level original FMA control leaves 8 values beyond the pre-registered FMA bound (NR 6, QC 1, BIR 1). The manager explicitly disposed these as a small DISCLOSED deviation under the standing FMA policy, with D6 deciding. Both failure counts remain in the packet. No tolerance or port arithmetic is changed.

MP52 remains compile-blocked. A 45-second CPU stack sample of its actual whole-program embedded kernel is inside recursive MLIR forward-slice construction, called by `CoalesceUtils.cpp:44`. See `MP52_COMPILE.md`; no successful compile, native E41 or forecast qualification is claimed for MP52.

Performance/memory: this packet changes no production launch, allocation, synchronization or compile path. The proof-only CUDA replay supplies the retained 1024-byte shared scratch as runtime-static plus launch-dynamic memory. Source/cubin hashes, actualGPU backend and attributes are sealed. Its eight-column timing is not a forecast-speed claim. Full final-composed GPU/E41/VRAM/compile and missing ice/radar schema qualification remain with their owners.

Reference repeatability: short compiler-ZERO controls are byte-identical for
mp50/mp51/mp8 across 3,285 paired fields. The full 24-hour c21 repeat is also
byte-identical: 25/25 complete output files and all 9,150 typed fields.
Both full runs have identical sealed inputs and the same admitted compiler-ZERO
executable. The duplicate second run was deleted after its hashes were sealed;
the first reference remains available. See reference_control/full24_zero_repeat.json.
