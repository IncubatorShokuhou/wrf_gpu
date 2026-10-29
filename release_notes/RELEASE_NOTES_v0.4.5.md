# v0.4.5 — regional studies and the graupel sedimentation fix

**29 October 2026.** The real-world checks pass with a disclosed final-frame
graupel trace disposition. The frozen A3 rule, timing and memory checks pass;
raw failures and the unresolved graupel trajectory cause remain visible.
No threshold or rule was changed.

[Known issues](../KNOWN_ISSUES.md) · [v0.4.4 notes](RELEASE_NOTES_v0.4.4.md)

## What changes

- **Regional calibration in waves.** Cloud, wind, temperature and humidity
  scores can be considered together, with non-degradation constraints,
  forcing/start contrasts, bounded waves, pause/resume and an external GO/STOP.
  The synthetic-observation twin harness tests the estimator; it does not
  establish real-station forecast improvement. See the [wave workflow](../docs/optimizer-waves.md),
  [cloud studies](../docs/optimize_clouds.md) and [joint objectives](../docs/optimizer_multiobjective.md).
- **Thompson graupel sedimentation.** Restore WRF's retained-entry `L_qg` gate.
  The correction is present in the native program actually loaded on the GPU.
  It changes the default forecast program; it does not resolve every trace-zero
  difference over a coupled forecast.
- **MYNN cloud PDF study options.** Modes 0/1 are CPU-qualified development
  options; coupled GPU forecast qualification remains pending. Default mode 2
  is unchanged. [Scope and restrictions](../docs/mynn_cloudpdf.md).
- **Safer restart admission.** Missing resume journals are handled on the error
  path without weakening coverage or authentication checks. Host reader-registry
  metadata is updated; these two recorded freeze exceptions preserve native
  forecast programs. Old checkpoint compatibility is not claimed.

**Still refused:** diffusion/damping parameter overrides, including `c_s`, and
the nine new soil/canopy/snow controls. Their public admissions remain REFUSED
until independent GPU qualification passes. Existing qualified surface controls
and experimental MYNN coefficients retain their published scope.

## Checks completed on the candidate

Model pin `254c482374`, source tree `827470e10f`. The original CPU-WRF is the
fidelity reference. Cache replay comparisons between GPU runs check regression
and determinism, not WRF fidelity.

| Check | Result and scope |
|---|---|
| RW1: warm cache | PASS; byte-exact short replays, zero new compiled artifacts |
| RW2: cross-date warm cache | PASS; WN3 dates and shifted start reuse sealed programs |
| RW4: bundle replay | PASS; three WN3 replay dates |
| Swiss 24 h | PASS against CPU-WRF; 25 frames and output integrity |
| WN3 0227 72 h | All three domains, 73 frames each; frozen A3 PASS, raw D6 FAIL, graupel integrity disclosure below |
| Compile cost | 489.9 s versus 473.420 s in v0.4.4: **+3.48%** |
| Same-session speed and VRAM/RSS comparison | PASS; fresh N2/O2 and N3/O3 pairs, numbers below |

Compile cost is the measured cold-minus-warm three-hour WN3 proxy, not cached
readiness or whole forecast wall time. The candidate used fresh-empty autotune
state; the previous-release baseline had seeded autotune state. This measurement
is not a like-for-like claim about compiler efficiency.

### Rain and vertical wind: the regular frozen A3 rule

The retained full-lead original CPU-twin floor is now an ordinary input to the
A3 check, rather than a release-specific rain waiver. The unchanged rule is
`C > limit OR C/X >= 0.75`. Coverage includes all 219 domain/frame pairs.
Five raw exceedances are FLOOR-LIMITED and zero are REAL FAIL: d03 RAINNC at
leads 69–72, plus d03 W at lead 69. At lead 72 the rain RMSE is 2.347817 mm
against the raw 1.0 mm limit; CPU-twin C is 2.111858 mm. Raw error has not been
relabelled as a raw-tolerance PASS. The original raw report remains FAIL.
The full reference contains 2,406 values, with its original pair provenance.

### Final-frame graupel trace: cause remains open

At d03 lead 72 the GPU QGRAUP field is zero while the original CPU-WRF carries
a trace (maximum approximately 6.58e-12 kg/kg). **The graupel fix is active on
the GPU. Pristine WRF on the tested reconstructed RC t72 columns also gives
zero.** Those hourly reconstructed columns are not actual pre-microphysics
savepoints and do not replay the final hour. The source of the differing
trajectory and the process that removes the trace between 71 and 72 h remain
**OPEN**. A coupled “trajectory trace, not a port defect” conclusion is not proved.

The investigation is documented in `v45-graupel` commit `340d8d8b56`,
`RC_TRAJECTORY_REPORT.md`. Closing the 71–72 h process and trajectory cause is
a **proposed v0.4.6 item**. This disclosure does not turn the raw integrity failure
into an automatic PASS or change a tolerance. The [manager decision](../docs/release/evidence/v045/MANAGER_DECISION_V045_QGRAUP_20261029.md)
explicitly accepts this disclosed trace disposition for v0.4.5, following the
v0.4.2 precedent and the new evidence. The [trajectory report](../docs/release/evidence/v045/RC_TRAJECTORY_REPORT.md)
is preserved from commit `340d8d8b56`. The raw integrity report remains unchanged.

The N72 command timer was 455.63 s. Its observed host RSS peak was 6.033 GiB
and sampled process VRAM peak 5,426 MiB. These are diagnostic observations
from the completed fidelity run, not accepted speed/memory ratios: that run's
watcher sample density did not pass. The fresh cohort must establish the
comparison against v0.4.4.

## Matched release-day speed and memory

The fresh same-session cohort contains N2/O2 and N3/O3, each 24 h on WN3 0227.
N is v0.4.5; O is v0.4.4. All four arms count under the frozen watch rule,
including continuous sample coverage; all four also pass the stricter 20/50
shadow watch. Every arm loaded the sealed GPU programs with zero new compiled
artifacts and no host suspend. No rejected earlier observation is pooled here.

| Metric | v0.4.4 O | v0.4.5 N | N/O and interpretation |
|---|---:|---:|---|
| Stepping, d03 segment median (s per forecast hour) | 5.7625 | 5.7425 | 0.9965; no resolved speed change, PASS |
| Process start to last output (s per forecast hour) | 6.8102 | 6.8593 | 1.0072; disclosed whole-run statistic, PASS |
| Mean sampled own-process VRAM peak (MiB) | 5,394 | 5,251 | 0.9735, PASS |
| Mean host peak RSS (GiB) | 6.0455 | 6.0425 | 0.9995, PASS |

For the primary stepping statistic, repeat spread S=0.0181 (1.81%); the
registered upper bound is **0.9965 + 0.0181 = 1.0146**, below the unchanged
1.10 STOP boundary. This does not establish a speedup. The two individual
stepping ratios are 1.0136 and 0.9797. Whole-run arm-window disk await stayed
below the registered 50 ms mean threshold, so no arm is excluded from that
secondary statistic. Memory means use only the same four counted arms.
The actual measured model PIDs bind 322–326 VRAM samples per arm to their GPU
assertions; other resident process samples do not set any of the reported peaks.

The first session ended before O2/N3/O3 started, and its N1/N2 watcher sample
density was insufficient. Those observations remain rejected. The host sampler
now uses absolute monotonic deadlines; the acceptance rule is unchanged. A
mandatory skipped arm now returns a nonzero session status.

O72 did not run. There is no same-session 72-hour old-release comparison; no
old-session or interpolated timing is substituted. N72 supplies the fidelity
check; the four fresh 24-hour arms supply the accepted speed/memory comparison.
[Receipt-bound release metrics](../docs/release/evidence/v045/R6_RELEASE_METRICS.json).

## Evidence and remaining limits

The release-preparation packet binds all numbers to SHA256 receipts from
`SHORT_VERDICT.json`, `RW_VERDICT_v045.json`, `FLOOR72.json` and the retained
R4 run metadata. Compiled hard-resource checks pass; compact-AOT optimized-HLO
metadata remains UNKNOWN where unavailable and is not reported as a measured
HLO PASS. CPU collection and packaging preparation are separate from GPU
qualification. `icloud=0` replay is checked, but no fresh CPU-WRF fidelity claim
is made because its original CPU twin was deleted. Nine-domain fidelity and
real-station calibration limits from earlier releases still apply.
