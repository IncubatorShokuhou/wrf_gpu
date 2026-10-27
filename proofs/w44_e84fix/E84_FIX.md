# E84 solar and ozone RC repair — worker gates PASS

**Scope:** nine solar and five ozone quotient families, in exactly
`coupling/physics_couplers.py` and `physics/wrf_cam_ozone.py`. Base source
`70fa838467`; measured arithmetic `cb98b44524`; final source tree
`70c5830686c69dd8185e2b30580cb22247465d7b`. The final source differs from
the measured arithmetic only in one helper docstring; source locations and
arithmetic AST are unchanged. The coupler file is byte-identical to the
measured version. Per-file patch IDs and SHA256s are in the
[gate packet](receipts/GATE_PACKET.json).

The two constant families use host REAL4 quotients. Dynamic sites use the
manager-authorized RN-emulation f64 island: REAL4 inputs, binary64 division,
then REAL4 rounding. The divisor stays opaque to reciprocal folding. Each
actual compiled site retains `div.rn.f64` and `cvt.rn.f32.f64`, bound through
the original protobuf caller stack and native thunk table to emitted PTX.
Root solar/ozone lead/60 sharing is proved by the actual optimized dependency
path. No custom kernel, State leaf, host transfer or host synchronization was
added. See the [S2 inventory](S2_ISLANDS.md) for operation/element counts.

## Scientific and lowering gates

All measurements below are **[M CPU]**, including CUDA-target CompileOnly;
no GPU execution or forecast validation is claimed.

| Gate | Result |
|---|---|
| 1, amended by manager 2026-10-25T18:09:44Z | All 504 product rows BYTE to unchanged base, eager and JIT; original frozen WRF bounds pass. 14,398,452 operand-range comparisons and 9,390,776 protected product quotient values over 677 real radiation clocks are RN-exact. Repaired rank surrogate has zero daylight/grazing/slope flips. |
| 2 | Independent original-WRF ozone profile reference: **0 ULP**, within the unchanged <=1 ULP bound, on all ranked cold columns and root radiation clocks. |
| 3 | Swiss d01, PROD d01/d02, WN3 d01 and fused d02/d03: original-protobuf HLO provenance, full-constant hashes and exact-thunk PTX binding PASS. Optimized native call/body censuses match the bound real-program inventory. |
| 4 | **14 deletion + 2 numeric semantic mutants**, each killed by a genuine AssertionError; zero setup errors in the accepted run. |
| 5 | Exactly two production files; per-file patch IDs; same-tool/input/options cost pairs below; all five E41 prescreens PASS, maximum stack **32 B**, no oversized kInput transpose fusion. |

The existing solar/ozone suite also passed **50 tests in 6.89 s**. Source
fixtures, reference asset hashes and tolerances were unchanged. The
[known-issues snippet](KNOWN_ISSUES_E84.md) preserves all 13 deferred families,
un-cleared physical/KF divisions, and the pre-existing CPU transcendental/FMA
differences. Those differences are identical in base and candidate.

## Matched cost evidence

Each pair uses the exact same harness bytes, case/input hashes, target and
autotune-0 options. Both sides pass E41. Nonfused pairs use the original
`f27cd1ec…` harness; the fused pair uses the corrected record-window wrapper
in `008d31a0c5`. The unrelated budget-lane autotune-4 counts were not reused.

| Program | Unique native kernels, base → repair | Static scheduled kernel references, base → repair | Planned temporary bytes, base → repair |
|---|---:|---:|---:|
| PROD d01 | 1104 → 1107 | 1506 → 1516 | 525,257,176 → 525,257,432 |
| PROD d02 | 753 → 751 | 1060 → 1064 | 642,901,760 → 642,901,760 |
| WN3 d01 | 1108 → 1111 | 1515 → 1525 | 525,544,168 → 525,544,424 |
| WN3 fused d02/d03 | 1227 → 1226 | 2163 → 2171 | 580,269,472 → 580,269,472 |

Argument, output and alias bytes are exact in all four pairs. The added
operations remain scalar XLA fusions plus the existing longitude field
quotient's f64 divide/narrow sequence; each site's scalar fusion count and
bindings are listed in the packet. There is no new persistent carry or
Pallas call. The longitude sequence adds conversion/materialization traffic;
planned peak scratch grows only 256 B on each root and does not grow on the
two child programs. That peak is compiler memory planning, **not measured
VRAM**.

**[I scenario, not GPU timing]:** charging each additional scheduled
reference once per radiation refresh at 2–10 microseconds/reference gives
20–100 us/root, 8–40 us/PROD child and 16–80 us/fused children. This explicit
assumption does not measure or bound the field-divide time. No s/fc-h,
cached-readiness or VRAM claim follows from these CompileOnly arms.

## Review and remaining release work

[Receipts](receipts/GATE_PACKET.json) include all five RN/ELF prescreens,
four unchanged-source baselines, resource counts, per-site scalar fusion
inventories, genuine mutation results, operand ranges and original-WRF
component comparisons. Exact optimized HLO/protobuf and compressed PTX/thunk
artifacts remain under `<USER_HOME>/wrf_gpu2_lanes/w44-e84fix/`; disposable
compile dumps and caches are removed after their consumers finish.

This branch is an RC composition item, **not a main merge before the RC
cut**. Composed-RC GPU E41/clock proof, RW1–RW4 including original-CPU-WRF
72-hour comparison, and actual GPU timing/VRAM/readiness remain required.
