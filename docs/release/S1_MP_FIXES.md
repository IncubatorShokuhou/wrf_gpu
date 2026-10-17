# Microphysics cloud number and radiation radii

| Configuration | Result | Qualification |
|---|---|---|
| mp=40, aercu_opt=0 | Wholly dry columns retain Nc=0; active columns perform WRF's final constant-droplet reset at every level | Pristine full-wrapper REAL4/REAL8 oracle; GPU/forecast validation pending |
| mp=18, use_mp_re=1, RRTMG LW/SW=4 | NSSL holds final cloud/ice/snow radii for the next radiation call, independently of radar diagnostics | CPU-oracle-qualified; GPU/coupled-forecast qualification pending |
| mp=24, use_mp_re=1, RRTMG LW/SW=4 | WSM7's existing radius diagnostics reach the held radiation carry | CPU-oracle-qualified; GPU/coupled-forecast qualification pending |
| mp=28, use_mp_re=1, RRTMG LW/SW=4 | Thompson aerosol calculates radii using its prognostic cloud droplet number | CPU-oracle-qualified; GPU/coupled-forecast qualification pending |
| mp=40, aercu_opt=0, RRTMG | Default radiation radii, as in WRF; no mandatory held-radius restart fields | Pristine compiled physics_init activation flags |

The existing GPUWRF_RRTMG_MP_RE switch still controls mediation. use_mp_re=0
or an incompatible LW/SW pair leaves radii inactive. Existing mp=8 behavior
is preserved. These changes do not admit additional nested configurations.

The mp=40 and WSM7 adapters pass WRF's dry TH_PHY into microphysics and
recouple its output to the moist theta_m carry using moist_physics_finish_em.
Pristine saturated mp=40 columns expose the old caller error; an independent
literal WRF arithmetic test covers WSM7's input and output boundary. WSM7 also
preserves the loader's disabled-radius selection through its adapter.

Radii are mass-grid REAL diagnostics in meters. Radiation receives the held
previous-microphysics values, converts to microns, applies WRF's land/ocean
and ice fallbacks, and retains the existing snow optics cutoff. Restart stores
all three active fields and refuses an incomplete active radius carry.

Evidence lives in proofs/s1_mpfix: the small tracked fixtures, source hashes,
caller forwarding seals, generation scripts, and tolerance manifest. NSSL
and WSM7 retain their existing pristine column oracle gates. The new Thompson
fixture includes both calc_effectRad and the final mp_gt_driver clamps.
The caller oracle compiles the literal physics_init and RRTMG mediation
blocks. An additional oracle executes the full unchanged radiation_driver
on three active real WN3 columns for mp=18/24/28 with use_mp_re=0/1
(18 calls). It uses Registry-default swint_opt=0 and the operand fixture
model top, 5000 Pa. This is a column/caller gate, not a coupled forecast.

The mp=40 fix adds a per-column activity reduction and conditional assignments.
Its measured five-column CPU artifact uses 127,424 temporary bytes versus
128,192 before; compile time is 7.74 versus 7.55 seconds. These CPU measurements
establish no GPU speed claim. The extra radius carry for each newly activated
scheme uses 12 bytes per mass-grid cell (about 15.7 MiB on PROD d02). NSSL
avoids radar work when only radii are needed. GPU overhead attributable to the
fixes and cached executable readiness remain unmeasured.

The final real WN3 mp=40 N2/N6 comparison passes all scored core D6 regions
and every child Nc region. Parent boundary Nc still fails: at the first step
the port activates one additional 44-level column (5,456 nonzero values versus
5,412 in original WRF). The maximum boundary error is 2.884308224e9 #/kg.
The final child post-RK Nc census records 790 negative visits, with minimum
-1,964,804.875 #/kg; their same-input WRF mechanism is unproved. Nested mp=40
therefore remains refused. No tolerance or mask was changed to admit it.

| MP | Real WN3 admission result |
|---|---|
| 18 | All N2/N6 numerical regions pass; parent step 2 retains 30 post-RK qi negatives (minimum -5.38e-23 kg/kg) and 46 Ni negatives (-3.03e-11 #/kg). Same-input WRF operator justification is missing. |
| 24 | All N2/N6 numerical regions pass; child post-RK qi retains 543 negative visits (minimum -1.04e-8 kg/kg), plus negative Ni placeholder values. Same-input WRF operator justification is missing. |
| 28 | Both domains fail initial QNWFA/QNIFA before stepping: maximum errors 1.690088e7 and 8.2910125e4 #/kg; normalized maxima 250.01 and 166.40 against the frozen bound 1. The arm was stopped after this decisive failure. |
| 40 | Child Nc and core D6 pass; parent boundary Nc and unresolved post-RK Nc negatives fail, as above. |

All four remain refused for nesting, with measured reasons in the admission
error and catalog. Evidence and original-reference hashes are bound in
proofs/s1_mpfix/NESTED_ADMISSION.json. Small negative amplitudes alone do not
establish the upstream mechanism or satisfy F2.

The pre-existing WSM7 density/thickness caller audit also remains open:
its adapter passes a dry-density EOS result as DEN, whereas WRF's microphysics
caller passes the dry moist_physics_prep_em 1./(al+alb) (not phy_prep's (1+qv)/ALT;
correction t-classd 2026-10-15), and its dz uses g=9.80665 instead of WRF physics g=9.81.
These inputs were not changed here; the column/radius qualification does
not establish complete literal caller parity.

NSSL radius-only CPU measurements on the active pristine 40-level column
retain 98,528 temporary bytes versus 99,104 without radii, with 504 more
output bytes. Warm medians are 0.7401 versus 0.7394 ms; compile times are
21.57 versus 20.87 s. These are component CPU costs, not GPU timings.

The corrected full-State dry-theta adapters add no measured CPU warm-time
loss in the small active-fixture probes. Mp=40 temporary storage changes
128,304 to 129,136 bytes (832 bytes for five 40-level columns), with compile
10.278 to 10.369 s and warm median 0.717 to 0.686 ms. WSM7 temporary storage
stays 24,992 bytes, compile changes 7.747 to 8.357 s and warm median 0.458 to
0.380 ms. Input/output sizes are unchanged. Different physical temperature
branches and shared CPU work prevent attributing these differences to speed
improvement; these measurements only bound the changed CPU artifacts.

Independent s1-gpuqual receipts use the exact physics candidate 3e70f3b4b,
the official GPU lock, and active radius mediation for mp=24/28. Each Swiss
42x42 case completed 24 h. The subsequent matched original CPU-WRF scores
fail mp=24/40 and leave mp=28 incomplete, as detailed below. Cold walls include concurrent CPU/IO work and
are not an S1 warm benchmark or a comparison against pre-fix performance.

| MP | Whole 24 h cold wall | Sampled peak VRAM | E41 disclosure for this experimental opt-in |
|---|---:|---:|---|
| 24 | 548.1 s | 3,334 MiB | STACK 3,864 B exceeds 256 B |
| 28 | 449.6 s | 2,868 MiB | STACK 928 B; kInput transpose up to 1,510 optimized HLO instructions |
| 40 | 501.9 s | 2,618 MiB | STACK 16 B passes; kInput transpose up to 2,226 optimized HLO instructions |

The transpose instruction bound is 500. These flags are disclosed under
wave S1's experimental-scheme policy; release defaults retain the hard E41
gate. Original receipts: s1-gpuqual/mp{24,28,40}/gpu_run.json and
e41/wave_gate.json under <USER_HOME>/wrf_gpu2_lanes/.

The matched 25-frame original-WRF science gate first fails RAINNC for mp=24
at hour 12 (RMSE 1.3609 mm, bound 1 mm), and for mp=40 at hour 13
(RMSE 1.9030 mm). Full field coverage is also false: mp=24 lacks QHAIL,
and mp=40 lacks 15 original fields including QNCLOUD/QNGRAUPEL/QNSNOW.
Mp=28 has no scored numerical failure, but seven missing original fields
(QNBCA, QNCLOUD, QNIFA, QNIFA2D, QNWFA, QNWFA2D, TAOD5502D) make its result
INCOMPLETE. These independent results do not establish the cause of the
rainfall errors and do not qualify any of these schemes for release.
Hashes and exact first failures from their grid_gate.json receipts are
preserved in proofs/s1_mpfix/EXTERNAL_GPU_RECEIPTS.json.
