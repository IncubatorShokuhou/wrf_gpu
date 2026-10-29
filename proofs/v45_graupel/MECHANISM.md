# WRF mp8 graupel entry gate

Status: a concrete operator defect is localized and patched. Its contribution
to the coupled N72 hour-72 event is **not yet measured**. No release integrity
exception or tolerance has been changed.

Pristine WRF 4.7.1 `module_mp_thompson.F` sets `L_qg(k)=.true.` when entry
`qg1d(k)>R1` (1915–1917). In the mp8 branch of sedimentation preparation
(3285–3303), it re-diagnoses working mass/number but only **clears** that flag
when updated mass is too small. The mp38 branch explicitly sets it true again
(3268–3269); that is not the mp8 behavior. Both fall-speed preparation (3744)
and the graupel fall loop (3902) require `ANY(L_qg)`. Once one entry cell survives,
the loop may also transport newly born graupel elsewhere in that column.

The port instead enabled fall speeds from the updated graupel field in every
column. Thus a column with no entry graupel could form graupel from rain freezing,
sediment it immediately, and lose the remaining trace to the **same** final WRF
`R1=1e-12` threshold. The final floor itself was not incorrect.

The fix retains an entry mask only on the mixed-phase WRF path, intersects it
with the updated positive-graupel cells, and passes its column `ANY` to the
fall-speed calculation. It applies to full-column and fused-preparation callers;
the mixed-phase WRF path remains explicit-only. The low-level implicit helper
accepts the optional mask for consistency. The mp38 implementation is unchanged.

Measured evidence (CPU; pristine `mp_gt_driver` is the scientific reference):

| Probe | Result |
| --- | --- |
| G01, retained N72/reference h60–72 | h71 CPU/GPU positive graupel cells 9/7; h72 3/0, CPU max 6.5775206462e-12. 56 reconstructed real columns saved. |
| G02, identical original inputs | 0/504 existing field-gate failures, 0 zero-pattern mismatches. Original h72 CPU traces survive in both operators. This excludes immediate erasure of those already-existing traces in this CPU replay. |
| G03, erase entry graupel | Both operators create 16 positive cells from rain. Erasing rain/number too gives 0 in both; input cloud ice and snow are exactly zero. |
| G04, restore only WRF's entry gate | Max graupel error falls 3.803381e-14 → 2.602085e-18 kg/kg. A rain-amount sweep stays in discrete freezing-table bins and produces no zero-pattern mismatch. |
| G05, density control near the existing floor | 129 explicitly modified copies of one real h72 column (density factors 1.16–1.20, reciprocal cloud/rain/number scaling; temperature retained). Old port erases 21 traces that pristine WRF keeps at 1.000252e-12–1.005569e-12; gate intervention preserves all 21. These are real-structured controls, not the original forecast operands. |
| Production patch | New tests: 4 passed, including pristine zero-pattern/increment checks, deletion sensitivity (21 erasures restored), old-byte controls with entry graupel, and fused gate wiring. |
| Existing regressions | 395 passed on the same immutable source, including mixed-phase and graupel WRF oracles, existing deletion controls, exit parity and the unchanged fp64 census. Completed IDs were retained across the bounded run/resume. |

The compact tracked fixture has 188 columns and SHA-256
`10663ff5e1c79fda1860bfbd0009f72d3beef904d1bd7df7b57b48fe102dfcc7`.
Its JSON manifest binds source artifacts and tolerances. G01 original reference
files were read only through facade copies, then those private copies were
removed; original references and retained N72 outputs remain untouched.

Oracle repair: the historical TH08 caller allocated only one HGT row, while
WRF initialization unconditionally formats `HGT(its+1,k,jts+1)` for a debug
message. GDB localized a larger-batch crash there. Providing only the missing
diagnostic HGT row changes none of the model equations. All 168 earlier G02/G03
columns and 13 output fields were byte-identical after that caller repair.

Efficiency assessment [I, source inspection]: the full-column path adds one retained boolean per
level, a column reduction and a mask of the existing graupel fall-speed activity.
It adds no host/device transfer, separate kernel launch or full-field floating
temporary there. The optional fused-preparation path passes one boolean per
column as an additional operand. Register pressure, compile time, memory and
speed effects remain unmeasured on GPU; `GPU_PLAN.md` preregisters those checks
and the required N72/D6/integrity attribution. No forecast-speed claim is made.

CPU qualification is sealed in `CPU_GATE.json`: 399 tests passed; 21,183 tests
collected with zero errors; mixed-phase OFF graphs (full-column and Python body,
dt6/18) are byte-identical after stripping locations; full and fused kernels
lower to Triton for sm120; the exact `night_seq_v9` CPU smoke passes, candidate
0 trace mismatches and deletion 21, with zero GPU processes. GPU-native E41,
timing/memory, N72 attribution and composed release gates remain pending.
