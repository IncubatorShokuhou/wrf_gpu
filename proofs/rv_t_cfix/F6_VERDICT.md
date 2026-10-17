# MERGE — t-cfix-pblmix F6

Reviewed head `b9b6891e5844175fd1928fd55cfea838a94e32c5`, base
`a0e061fd9d13afbf71b42b3f0b555d1e95b8b615`. CPU source/oracle/caller acceptance;
GPU coupled-forecast qualification remains required for c03/c04/c09/c25.

[M CPU] Independent critic results:

- CF01: 49 PASS, 0 failures/errors, 106.69 s (all 28 producer gates plus changed
  YSU/cuplus forwarding consumers).
- CF02: rebuilt unmodified pristine public MRF, BL_GFS and MYJPBL calls on the
  frozen 16/18/16 cloudy CPU-WRF columns. Source and input-text hashes verified;
  all nine DOUBLE/REAL/with-ice/without-ice dumps reproduce exact SHA and values.
- CF03 + CF06: seven two-step operational caller cases PASS. Actual adapters,
  kernel-rate sentinels and an isolated RK consumer establish once-only qc/qi
  application, retention of a separate dynamics increment, and correct next-step
  operands. Covers MRF RK, GFS/MYJ legacy, MYJ cu7 rate path, ice/no-ice where
  admitted. This is wiring evidence, not a numerical dycore/forecast oracle.
- CF04: both CPU native-facade default traces EXACT:
  Swiss `f672a5739cb96e388b6b7a40ef80337bf81bdc14ef15268f3476e09d79fe055d`;
  PROD d01 `6fc715a2bd630934c4cf0f94427df19a527cbf429011c9ba5efe4d712744d27b`.
- CF05: independent in-memory removal of MRF's ice lane gives one genuine
  numerical AssertionError, zero setup errors (4.60 s). Producer's eight mutation
  definitions/receipts also inspected. Initial critic guard-order setup failure
  preserved separately as INVALID and never counted as a kill.

Source matches pristine pbl_driver condensate arguments and addtendc tendency
outputs. MYJ uses CWM=QCW+QCI and species row 4; GFS preserves its specific-humidity
recovery; MRF returns its already-solved ice RHS. Registry QI slot selection is
static. MRF's UST INOUT rebuild and GFS's zero-based KPBL correction are verified
by the same original public-call oracle. Existing RK `_physics_sc_outside_spec`
enforces WRF lateral ownership; `ustar` survives the non-dry replace list. No new
State/restart fields, baked assets, or changes to unrelated scheme kernels.

Efficiency [I]: MRF/MYJ already solved a zero ice RHS; added condensate inputs and
rate arrays increase traffic/liveness. GFS adds one one-RHS Thomas forward/back
scan and its full-column temporaries; sharing qc/qi elimination through the
existing two-RHS helper is a possible later optimization. No measured GPU launch,
memory, readiness or speed claim is made. Default programs are unchanged by CF04.

Scope: documented surface proxies in the real-column fixtures do not close the
separate F4/F12/F13 staging/handoff findings; legacy Euler→RK conversion is the
separate F9 lane. Compose those fixes while retaining the qc/qi raw rate keys,
then perform the affected GPU D6 runs. This verdict clears only the posted F6
series; it does not clear all caller-audit findings or the release.

Artifacts: `<USER_HOME>/wrf_gpu2_lanes/rv-t-cfix/F6/CF01*` through `CF06*`,
`rebuild_oracles.py`, `test_two_step.py`, `run_mutant.py`. No GPU used; cache off.
Disposable oracle builds and immutable snapshot removed after their last use.
