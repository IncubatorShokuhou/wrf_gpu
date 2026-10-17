# MSKF, KF-CuP and CAM Zhang–McFarlane

**Status in v0.4.0:** MSKF (`cu_physics=11`) and KF-CuP (`cu_physics=10`) are **validated 24 h**: each passed a 24-hour comparison with the original CPU WRF in the v0.4.0 release check, in one tested single-domain configuration. Other combinations and nested domains are not validated. CAM Zhang-McFarlane (`cu_physics=7`) is **experimental**: it passed one 24-hour comparison outside the release test set, but its compiled GPU code exceeds the code-quality limits.

`cu_physics=11`, `10` and `7` select dedicated ports of WRF 4.7.1
`module_cu_mskf.F`, `module_cu_kfcup.F`, and `module_cu_camzm(_driver).F`.
They are also checked on the CPU against the original WRF code. GPU timing
and memory use have not been measured. They are not aliases of KF-eta.

The supported caller profile uses Thompson (`mp_physics=8`) and
RRTMG (`ra_sw_physics=4`, `ra_lw_physics=4`). MSKF and KF-CuP require MYNN
PBL and surface layer (`5/5`). CAM-ZM accepts CAM-UW with MYNN surface layer
(`9/5`) or MYJ with Janjic surface layer (`2/2`). The retained Noah-MP adapter
selects MYNN surface internally, so CAM-ZM/MYJ with Noah-MP is refused.
Other microphysics, radiation and producer combinations are refused with a
reason until their interfaces and combined behavior are validated.

Both native/nested and daily replay loaders bind these selectors and controls.
Python operational callers must select active physics and WRF RK source
coupling (`rad_rk_tendf=1`). Tendencies are mass coupled and consumed at the
RK stages; they are decoupled after consumption. Cloud countdowns, held
momentum/moisture/number rates, rain, cloud diagnostics and source-event
counters survive checkpoints. Bad or missing restart fields fail closed.
The daily host loop selects the full operational carry for these options,
preserving its global clock and held fields across hourly and auxhist segments.
Daily checkpoints use the complete carry codec, and sticky CAM entropy failure
reaches the selected stepper's boundary guard.

KF-CuP honors `numBins`, `thBinSize`, `rBinSize`, `minDeepFreq` and
`minShallowFreq`. WRF's Registry default for **every one is 1**. An example
explicit PDF configuration is `5, 0.1, 0.0001, 0.1, 0.001`, respectively.
The port currently supports positive odd bin counts, positive finite bin
sizes and frequencies in `[0,1]`. Aerosol/chemistry coupling is refused:
use `shcu_aerosols_opt=0` and `aercu_opt=0`. Original KF trigger 1 is supported.

CuP's shallow/deep cloud fraction and temporary condensate radiation views
follow the original radiation driver; `shallowcu_forced_ra` is bound.
MSKF retains default-on original CMT and WRF's forced `icloud=1` and
`cu_rad_feedback=True`. Original REAL arithmetic remains REAL; the genuine
MSKF CMT and CAM DOUBLE islands remain DOUBLE. W0AVG and cloud lifetimes are
resident, and active clouds skip fresh parcel solves within each column.

Exactly calm convecting MSKF columns have an upstream `VCONV=0` singularity.
The port preserves its non-finite results and reports them through the census
and finite guard. It does not introduce a wind floor or a finite replacement.
CAM's original entropy-inversion fatal status is sticky until an existing host
output boundary reports it. See the source findings and measured compiler
deviations in [UPSTREAM_WRF_BUGS.md](../proofs/v040/cuplus/UPSTREAM_WRF_BUGS.md).

Evidence includes independent full original callers, active deep/shallow
soundings, rotated nonzero transverse winds, actual 44-level Swiss forecast
profiles, and repeated cloud-lifetime calls. The Swiss input bank is assembled
from original history with explicitly controlled missing caller diagnostics;
it is not a captured full-forecast physics savepoint. Literal checks
use the port's equations with REAL/libm rounding against independently compiled
original WRF. Small compiled deviations remain documented; no failed strict
column check is described as a pass.

A short CPU run on a real 44×42×42 Swiss case completes three steps for all
four supported pairings. Native checkpoint continuation preserves every carry leaf byte for
byte: CuP/MYNN 202, MSKF/MYNN 187, CAM-ZM/CAM-UW 202 and CAM-ZM/MYJ 157.
All 66 main guard counters and CAM source events are zero. Tiny option-only
post-RK negatives are preserved and quantified in
[F2_DISPOSITION.md](../proofs/v040/cuplus/F2_DISPOSITION.md); this short CPU
run does not show that negative values are absent in general.

The compiled CuP kernel shows no double-precision (fp64) operations in its
optimized code. On CPU its
40-level five-bin caller compiled in 16.19 s, with 72,464 temporary bytes;
median fresh/held calls were 6.91/0.44 ms. These are CPU interpretation costs,
not GPU timings. Each device column owns its held-cloud branch inside one
Pallas program; no host transfer is added inside the timestep. Original CMT
and CAM DOUBLE work remains explicit. The default forecast program is
unchanged.
