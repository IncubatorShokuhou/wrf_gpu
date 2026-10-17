# AOT value-key audit (t-aotkey)

This change hardens executable selection. It changes no physics, dynamics,
kernel, or operational-step arithmetic. CPU key/ABI checks are regression
evidence, not WRF fidelity evidence.

The original `fn_identity_hash` used code but omitted bound defaults and closure
cells. Fused cascades close over **whole** namelists: their metrics,
`radiation_static`, and split Noah-MP case children are constants there even
though ordinary domain steps receive them as operands. The synthetic fused aux
covered only the namelist aux. Actual B/Z Noah parameter bundles in an ordinary
namelist were already content-keyed; that was not the missing determinant.

Other confirmed gaps were keyword pytree static metadata, a second positional
argument unconditionally treated as the namelist, `_StaticHolder` subclasses,
and module/default/host-loader table values outside the namelist. Concrete
array materialization failures previously weakened to shape/dtype. Read-only
P3 mapping proxies also fell through to truncated array reprs. These now use
content, or refuse cheap reuse when a concrete value cannot be read.

`GPUWRF-AOTKEY-v5` invalidates the old metadata addresses. Exact-HLO/JAX cache
entries still include the actual constants. The new key preserves date-blind
clock aux, runtime operand values, and namespace-only terrain provenance rules.
Changed values baked into fused closures must now compile separately; passing
those values as actual operands could recover cross-case reuse in later work.

External-reader coverage is pinned in
`tests/test_aot_baked_constants.py::_EXTERNAL_READERS`. Land/urban/RUC parsed
bundles use static holders, CLWRF clock/gas data use traced operands, and active
Thompson/aerosol, RRTMG, surface tables, KF, RRTM, CAM, Goddard, NSSL and P3 tables
use actual bound/cached values. A new reader requires an explicit disposition.
The composed feature-wave audit covers 23 reader modules, including UrbanInputs
as traced namelist data (content-keyed when closed over). NSSL aliases share
resolved REAL4/REAL8 namespace snapshots, rather than only a JSON file digest.
Initialization includes the RRTMG solvers' host cloud/coefficient tables as well
as their bound defaults. Source edits and resolved import-time switches retain
their existing guards; CAM_TILE_COLS is now registered too.

After the glacier-main rebase, QNSE surface tables and cached surface/PBL
constants are keyed for the active QNSE pair, and GFS shallow lookup tables for
SAS94/95. Both REAL modes use the actual cached values. Inactive defaults skip
these providers. The composed affected gate passes 84 tests (AK21), including
eight new mutations and the complete reader/import-env censuses. Four provider
tests pass full cheap-key, lowered HLO and actual output checks together (AK22).
Restoring only the old table-hash function yields four genuine call-phase
assertion failures, zero setup errors (AK24).

Current main `d84b3699d` subsequently added the shared CuP/MSKF import-bound
tables. Their seven actual array/scalar fields are keyed for cu=10/11. The final
23-reader/scanner gate passes 45 tests, including 14 full-key/HLO/output field
mutations (AK26). Deleting the new provider hashing produces two genuine
call-phase assertion failures, zero setup errors (AK27). These cache regressions
make no coupled-forecast or WRF-fidelity claim.

Changed table-key component costs, CPU only: QNSE first 5.107 ms, warmed median
1.433 ms, combined incremental host peak 1.25 MiB; GFS first 5.942 ms, warmed
0.724 ms, peak 1.75 MiB (AK25). CuP first 2.848 ms, warmed 0.241 ms, peak 1 MiB;
MSKF reuses the already loaded data (AK28). These measure `baked_table_hash`, not
the whole key, production memo, timestep, GPU or readiness. The existing hot
memo and fused executable cache are byte-identical to pinned main `d84b3699d`.

Quarantine markers preserve the first rejection, including both HLO digests;
later attempts still remove stale payloads without replacing that witness.

Measured CPU component cost on matched parent/candidate canary metadata, fast
defaults ON: memo lookup median 152.96 -> 156.61 us; full-key median 2.986 ->
155.063 ms; first key 1.312 -> 2.233 s. Peak setup RSS growth differed by 126 MiB;
the warmed calls added no peak RSS. These are host component numbers, not S1 or
GPU-readiness measurements. No new steady-step work is added behind the existing
memo. The added table snapshots are startup/signature costs.

## Native E1 diagnosis

The measured native evidence, all 26 first-step field deltas, primary compiler
schema sources and ALISIOS guidance are in [NATIVE_E1.md](NATIVE_E1.md), with
hashed receipts in [NATIVE_E1_RECEIPTS.json](NATIVE_E1_RECEIPTS.json). The probes
used the actual original 86e630370 source, original case/tables and read-only
retained binaries. No HLO, target or ABI guard was bypassed. The requested
20-minute GPU cap was respected for each arm; completed disposable caches and
snapshots were removed.

Fresh compiled/export/reloaded execution is byte exact for 183 leaves at
steps 1 and 67. The same-HLO retained native realization differs in 26/98
leaves, with changed FP32 reduction SASS and matching CUDA 13.2.78 toolchain.
Copying current original tunes does not reconstruct that retained realization.
An exact serialized-options replay is rewritten by the backend: requested
memory zero becomes 8.416 GB, so historical memory-budget causality is not
eliminated. E99-style native-codegen variability is supported; historical E1
closure and WRF faithfulness of both trajectories remain unproven. One sealed
compiled bundle is the demonstrated byte-exact control mitigation.

Current-main 7cd23caef composition on frozen f92731e9f passes nine targeted
checks, covering the expanded 29-reader/environment census and actual
QNSE/GFS/CuP key/HLO/output mutations (AK52). Earlier genuine deletion controls
and numerical validations retain their exact original source scope.

CPU receipts in `<USER_HOME>/wrf_gpu2_lanes/t-aotkey/` include the 163 passing
broader consumers (AK06), the final scanner closure (44 pass, AK09), the table
delta (8 pass, AK11), and six genuine intended assertion kills across five
mutation classes (AK13b; zero setup errors). The one earlier scanner failure was
reproduced on the unchanged baseline and fixed. Both established release jaxpr
identities were exact on the pre-glacier source: Swiss `2bf12cb8`, PROD
`6fc715a2` (AK12).
On the later composed source: 105 cache consumers pass (AK14), all 45 affected
reader/flag/scanner checks pass (AK15), the NSSL alias mutation passes (AK17), and
five clean-archive checks pass (AK19). Both identities also remain exact after
actual metadata-key setup (AK16/AK18), with x64 configuration preserved.
Those receipts predate the approved glacier fix on main `0821ebc0a`. The current
Swiss gate is `f672a5739cb96e388b6b7a40ef80337bf81bdc14ef15268f3476e09d79fe055d`;
PROD d01 remains `6fc715a2`. AK23 establishes the new full Swiss identity after
actual non-null v5 metadata-key setup, with x64 preserved. The final CuPlus
composition has a separate source-bound identity receipt.

The final source-bound packet is `COMPOSITION.json`: Swiss full `f672a573...`
and PROD d01 full `6fc715a2...` both match after actual non-null v5 key setup,
x64 True -> True (AK29/AK30, 36.3/66.3 s CPU traces). Only three runtime cache
files differ from the pinned current model main. Previous receipts are retained
with their original source scope. Own detached snapshots and pytest/AOT caches
have been removed; original ALISIOS artifacts remain untouched.

At 2026-10-14 03:23Z the expected lane lease file did not exist. No GPU run has
been attempted. The E1/third-trajectory lane gate stays OPEN pending native
evidence; this is not a global E1 fix or a release qualification claim.
