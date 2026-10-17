WRF MYNN cold-start undefined RMOL, exposed by P3 (2026-10-14)

The read is located. All four requested full references are complete. c21/c22
pass the common bounded fields against compiler-ZERO original WRF; full schema
coverage remains incomplete. No numerical port change has been made.

In upstream `phys/MYNN-EDMF/module_bl_mynnedmf.F90`, `mynnedmf` declares local
`rmol` at line 473 and passes it into cold `mym_initialize` at line 719. Its
first assignment is later, at line 879. `mym_length` CASE(1) reads it in the
stability branch at line 1842 and in the length expression at lines 1843/1845.
CASE(2) has the corresponding read at line 1990.

TP30 Valgrind on the actual original serial executable reports that undefined
conditional in `mym_length`, called through `mym_initialize`, with its origin
in the `mynnedmf` stack. TP31b maps instruction `0x6714c7f` through the measured
PIE load bias `0x4000000` to preprocessed line 1822, upstream line 1842. The
raw module rebuilt with line information has the same complete `.text` SHA
`aff0edaa9c029eebac45f30a487b6a9d6a5f0f0a5660a348b5891b7923a0047e` as the
original object. The actual executable's eight instruction bytes match too.
The initial TP31 mapping omitted the PIE bias; its unresolved `??:0` result
is not admitted. TP31b reuses the compiled object and rejects empty windows.

The port already uses zero for this exact input in
`mynn_coldstart_init_columns` (`src/gpuwrf/physics/mynn_pbl.py:1150`). This is
the standing compiler-zero convention for an undefined upstream local. An
ordinary build does not guarantee a zero first-touch stack. The old v014/BP36
component proofs already documented this site; the new evidence exposes its
material coupled P3 symptom.

TP25 executes the same original serial WRF binary, working directory, inputs,
namelist and environment twice for Swiss mp50. All 366 initial fields are byte
equal. At 18 seconds only 355/366 fields match: maximum differences are
0.576355 K in T and 149.99999 in QKE. At 36 seconds T differs by 1.174866 K.
The original executable SHA256 is
`52f914cb38e0899d8a1da54bd52f3e08d81ba0f6a7015d3159e48393f5226bd6`.
Its private copy changes only two diagnostic filename literals.

TP25b repeats release-default mp8 using the same original build and private
copies of the retained original-WRF Thompson tables. All 362 fields are byte
equal at 0, 18 and 36 seconds. This observes a dependence on configuration;
it does not locate the defect inside P3. The QKE difference is concentrated in
the lowest 14 levels. MYNN's cap at `MYNN-EDMF/module_bl_mynnedmf.F90:3143`
explains the value 150 but is not evidence of the undefined read's location.

The private active-physics compiler-zero build, FP traps off, passes TP27's
3,285 paired field comparisons: mp50 3×366, mp51 3×367, default mp8 3×362 at
0/18/36 seconds, six successful runs, all floating fields finite. Inactive
Morrison mp40 retains original compiler flags because its blanket SAVE and
automatic arrays reject initialization under GCC14; it is never selected.
This short control alone does not prove full-run repeatability. Full
initialization spread and retained-GPU D6 scores are recorded below; tolerances
are unchanged. The full c21 repeat is also BYTE: 25/25 whole files and all 9,150 fields.
Material initialization dependence is disclosed under the standing experimental
policy, rather than treated as a physically defined pristine value.

External immutable receipts: `TP25_same/receipt.json`, `TP25b_mp8/receipt.json`,
and `TP29_population.json` under `<USER_HOME>/wrf_gpu2_lanes/t-p3fix/`.
Source localization: `TP30/memcheck.log`, `TP31/receipt.json` (text identity
only), `TP31b/receipt.json` (accepted source map), `TP27z/receipt.json`.

TP32/50a completed the full 24-hour c21 zero reference: 25 finite frames,
WRF_SUCCESS, 502.373 s including output verification (WRF 499.13 s, peak
RSS 225,260 KiB). All three namelist/IC/BC hashes equal the original sealed
c21 run. TP33c scores the retained real GPU against those frames with frozen
manifest SHA `9df4fab8514523286f07b127eb8bf57a7545459c4448fef721ac44c0ecc351dc`:
zero bounded failures, versus 92 against the original raw-init reference.
This is numerical qualification of the common bounded fields, not a complete
D6 pass: the old GPU output omits D_ICE, QIB, QIR, REFL_10CM, RHO_ICE and
V_ICE. The scorer therefore returns INCOMPLETE. Full c22–c24 zero references,
original-reference spread, and full-run repetition remain pending.

Full c22 now also completed: 25 finite frames, WRF_SUCCESS, 614.569 s including
verification. Its retained GPU has zero bounded D6 failures against ZERO,
versus 142 against raw init. Seven missing GPU fields retain INCOMPLETE:
the c21 six plus QNCLOUD. No numerical P3 port fix is justified by these runs.

The full 25-frame ZERO-vs-retained-original CPU comparison reproduces all
92 (c21) and 142 (c22) bounded misses. This is empirical initialization
sensitivity between two conventions/executions, not a statistical error bound.
Original-vs-ZERO RMSE is 1.932395/1.407941 K for T at h1, 3.134493/5.504015 K
for T2 at h24, and 4.566383/14.283764 mm for RAINNC at h24 (c21/c22).
The corresponding GPU-vs-ZERO h24 RAINNC RMSE is only 0.240275/0.250673 mm;
maximum hourly T RMSE is 0.062892/0.065661 K. Both arms of the sensitivity
comparison have 25 finite frames and identical sealed namelist/IC/BC hashes.

P3 configurations therefore depend materially on an upstream undefined value
and require the standing experimental disclosure, even though the common
bounded fields pass against the defined zero convention. Blanket compiler
initialization changes all eligible local variables; the exact Valgrind/PC
proof identifies RMOL, while additional undefined sites are not excluded.
Source bodies and all production port files remain unchanged. The complete
field-by-hour sensitivity reports are TP35_spread/{c21,c22}/D6_d01.json;
compact source-bound numbers are initialization_spread_c21_c22.json here.

All four requested full references now exist: c21/c22/c23/c24 each have
25 finite hourly frames (100 frames, 36,875 typed field-frames total).
ZERO_REFERENCE_INDEX.json binds their immutable receipts and output hashes.
Their original-vs-ZERO sensitivity produces 92/142/82/118 bounded misses,
respectively, with no nonfinite fields. This does not alter D6 bounds.
c23 retains only a GPU initial frame (0 bounded misses, INCOMPLETE);
c24 retains no GPU output (INCOMPLETE, comparison unavailable).

The native retained c21 cubin replay additionally preserves 240 strict
column deviations. A new SAME-INPUT 44-level original WRF FMA control
reproduces the pre-registered P348 spread exactly. Eight values remain outside
that bound: six NR, one QC and one BIR. The manager explicitly accepts these
as a small DISCLOSED deviation under the standing FMA policy; D6 decides.
No P3 port change is made for them. Native strict and derived-bound failure
counts remain visible in native_fma44_final_score.json.

The full 24-hour c21 ZERO repeat additionally proves 25/25 complete files and
all 9,150 typed fields BYTE across two successful finite runs. First reference
preserved, duplicate outputs deleted after hashing; full24_zero_repeat.json
records both source/input/output manifests.
