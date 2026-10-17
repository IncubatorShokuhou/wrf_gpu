**MERGE `7672228115299cb2a5f63bb63f3cd660e0caf377` for the manager's 05:16Z GPU gate.**

Source/CPU review complete; no remaining source/CPU blocker. This does not
declare actual GPU E41, trajectory BYTE equivalence or timing passed. Those
checks explicitly follow this verdict under the manager's gate plan.

Product delta versus main db78f9bd8c is confined to physics_couplers.py:
the unchanged driver re-formation now runs under `lax.cond(any(beyond))`;
its rho/qv/p_hyd/Exner operands are read inside the branch at native level zero.
The in-range branch returns the existing handles. The original PSFC seam is
preserved. No shared State flux mutation, new persistent carry or explicit
host synchronization API is introduced. The scalar predicate remains a device
value, not a Python boolean.

Independent evidence:

- [M CPU] **13 affected tests PASS**, 37 unchanged deselected: finite mixed
  extreme fluxes, in-range BYTE pass-through, branch placement, cu3/cu6/cu16
  input preservation, PSFC/previous-EL producer contract and two-step caller.
- [M CPU] Compiled candidate versus compiled parent db78 helper, with the same
  unchanged dependencies: **9/9 flux arrays BYTE, 0 ULP** across in-range,
  mixed two-of-six beyond limits, and all-beyond cases. State physical/shared
  handles unchanged. This is CPU regression evidence, not a WRF or GPU claim.
  The initial probe setup error (removed parent import) is retained separately;
  it supplied no numerical evidence.
- [M CPU] M7 (re-formation outside the conditional) and M8 (`all` instead of
  `any`) each fail their intended assertion, zero setup errors. M8's mixed
  columns make the branch decision matter. Snapshot source restored afterward.
- [M source] Physics formulas, MYNN kernels and pristine oracle fixtures are
  unchanged; existing oracle/fidelity evidence is reused. D6 5/5 for F20 is
  the manager's existing result, not a newly executed review run.

[M artifact audit] Independently re-scanned seven retained optimized-HLO/native
artifacts, verified their SHA against receipts, and checked every retained
kernel for STACK/nonleaf/callgraph failures. Calibration reproduces the GPU
F20 failures **803/694 HLO, STACK152**; pre-F20 is clean, max kInput407.
The five final C5 prescreens are clean:

| stepping program | max kInput HLO | max STACK B |
|---|---:|---:|
| PROD d01 | 407 | 152 |
| PROD d02 | 360 | 32 |
| WN3-0227 d01 | 407 | 152 |
| WN3-0227 fused d02+d03 | 400 | 32 |
| Swiss d01 | 346 | 16 |

All have zero nonleaf-over-1k and unknown callgraph entries. Swiss's largest
transpose of any kind is 364; the table specifically reports kInput.
Actual GPU executable scans remain authoritative.

[M rich GPU stack frames] The 803 body includes Noah soil_thermo181 and
water_hydro170 instructions; the 694 body includes Noah energy458 and
soil_thermo74. Neither has an observed innermost limit-helper frame.
266/86 instructions respectively have no source frame. These are mixed
multi-output fusions, not a limit-helper runtime attribution. The worker's
pre-F20/control bisection plus calibrated artifact results establish that the
new view triggered the compiler choice; reverting w faces did not resolve it.
Output-only and other fences were tried and failed before C5; no threshold change.

Prescreens were compiled on 4fab1d93a4. Its coupling module is BYTE-identical to
7672228115. The rebase changes only cu6 Tiedtke and opt-in NTU mp56 source, outside
the selected default mp8/cu1-or0 paths. This scope is recorded explicitly in
SOURCE_REUSE.json; it is not claimed that every source file is identical.
The post-verdict GPU gate must use the exact approved head.

Independently reproduced official native-facade default jaxpr hashes [M CPU]:

- Swiss: `a6166b20e08453f7a6d72167807c22610bf0453aebe17fc663d48633fff40cef`
- PROD d01: `4e1879c2b1f1581867fa6d3862a0712a7477433880c4deb6a193574818d20865`

Performance/memory: [I source] the usual path avoids re-formation and its live
column-layout operands, adding a 2-D bound test/reduction and conditional.
The rare branch computes a hydrostatic profile separately; no new State carry.
Static compiled kernel counts include the untaken branch and cannot establish
warm runtime or peak allocation. [U] GPU host syncs/transfers, live memory and
warm per-root timing remain unmeasured by this critic. Compile samples are not
a speed claim, and altered fusion/FMA boundaries can make whole-GPU frames non-BYTE.

Next, exactly as manager 05:16Z: v-prep runs PROD/WN3-0227/Swiss 3 h BYTE
regression against the retained F20 e7e37390cc frames, whole-program GPU E41
(including child/fused programs), and matched warm per-root timing/no added
host syncs. Any non-BYTE case requires its full 24 h D6 against original CPU-WRF.
No fidelity or performance waiver. The prior GPU outputs are a regression
reference, never scientific truth.

Own snapshot/pytest/cache cleanup complete; worker-owned retained HLO files
were read only. Reader DONE: their owner can delete the temporary dumps now.
Old hashed decisions are preserved; this is a separate sealed C5 verdict.
