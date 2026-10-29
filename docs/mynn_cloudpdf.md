# MYNN cloud PDF options

CPU-qualified development option; coupled GPU qualification is pending.

`&physics bl_mynn_cloudpdf=0` selects WRF's original Sommeria–Deardorff PDF;
`1` selects Nakanishi–Niino (2004). The existing default `2` retains
Chaboureau–Bechtold. This is one scalar shared by all nests, as in the WRF
Registry. A selection changes the compiled program and may compile cold.

Cases 0/1 change MYNN cloud production, its buoyancy feedback and the clouds
available to radiation. Use [icloud_bl](radiation_pbl_clouds.md) to control
only the radiation merge, or [bl_mynn_edmf](mynn-budgets.md) to disable plume
mass flux. These are different WRF experiments.

Case 0 carries TSQ/COV as well as existing QSQ; case 1 carries SH3D and uses
the existing previous-call EL_PBL. Selected moments are initialized by the
WRF five-pass cold-start procedure, updated after MYNN and retained through
the RK step. Native restart readers require the selected moments, so an old
CASE(2) restart cannot silently become a CASE(0/1) forecast.

The public choices are 0/1/2, with MYNN5. Cycling, stochastic physics and
MYNN budget histories with 0/1 are not qualified and are refused. Negative
PDF selections remain unsupported. Both native and legacy column paths
contain the PDF formulas. Full GPU/nested forecast qualification is pending;
the numerical evidence packet records the precise completed CPU scope.
The [review report](../proofs/v45_mynnopts/REPORT.md) includes the fresh original-WRF
recapture, compiled rounding guard, runtime/restart tests and default identity proof.

Case 0 uses WRF's explicit DOUBLE variance island (`t3sq/r3sq/c3sq`), not an
fp64 fallback. Extra persistent logical storage is two REAL profiles for
0 and one for 1: `8*nz*ny*nx` and `4*nz*ny*nx` bytes per domain. At PROD
d01 120×70×44 these are 2.82 and 1.41 MiB. Moments stay resident during the
timestep; checkpoint and snapshot payloads can grow by these bytes. No extra
output-time physics solve is added. Actual native latency, kernel count, scratch,
peak VRAM and compile/readiness cost remain unmeasured; see the
[preregistered plan](../proofs/v45_mynnopts/GPU_PLAN.md).

This feature makes additional WRF configurations available. It does not
establish improved cloud or station skill.
