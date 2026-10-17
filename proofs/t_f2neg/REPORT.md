# Pristine-WRF F2 post-RK comparison

**Result: original WRF reproduces the reported large intermediate-negative magnitude and interior-location classes. A blanket amplitude-bound or every-event F2 closure is not established.**

[M] Both 24-hour observed forecasts completed 4,800 dt=18 s steps. All 18,125 original output field/frame pairs are byte-identical to uninstrumented serial WRF (default 9,050; WDM6 9,075), with zero nonfinite output or post-RK observations. No gpuwrf numerical source changed.

The default QC and Ni extrema occur at the same cells in WRF and GPU, separated by five and one time steps respectively. Default QS is at the same cell one step apart. The reported values are uncoupled kg/kg or #/kg, before microphysics. They cannot be described as purely roundoff.

## release_default

[M] Post-RK visits: WRF **7,622,386** (7,255,270 interior); GPU **7,776,293** (7,391,406 interior). These are phase-field visits, not unique physical events. All observed species have 4,800 observations. QV is observed with zero negatives. Fields absent in WRF are not credited as observed zero.

| Field | WRF minimum | GPU minimum | abs(GPU/WRF) | WRF negative visits | GPU negative visits |
|---|---:|---:|---:|---:|---:|
| qc | -5.3879048e-06 | -5.1977713e-06 | 0.964711 | 1,240,994 | 1,272,746 |
| qr | -3.2871779e-07 | -3.0774171e-07 | 0.936188 | 573,467 | 589,612 |
| qi | -7.8355065e-08 | -8.9356874e-08 | 1.140410 | 1,094,357 | 1,108,044 |
| qs | -6.9883199e-06 | -6.7802184e-06 | 0.970222 | 1,356,351 | 1,372,248 |
| qg | -1.5135541e-06 | -2.1104718e-06 | 1.394381 | 293,118 | 301,237 |
| Ni | -15129.827 | -14834.351 | 0.980471 | 2,350,421 | 2,398,815 |
| Nr | -17.355766 | -15.36557 | 0.885329 | 713,678 | 733,591 |

Observed original-WRF wall: 1134.140 s, concurrent nice-19 CPU run including initialization and diagnostic I/O. This is not an S1 benchmark or an observer-overhead measurement.

Full CPU all/ring/interior quantiles and WRF values at the exact GPU-minimum step/cell are in the accompanying JSON. GPU quantiles were not captured by v-run and are unavailable; no quantile-equivalence claim is made.

## mp16

[M] Post-RK visits: WRF **13,050,108** (12,614,276 interior); GPU **16,758,185** (16,292,064 interior). These are phase-field visits, not unique physical events. All observed species have 4,800 observations. QV is observed with zero negatives. Fields absent in WRF are not credited as observed zero.

| Field | WRF minimum | GPU minimum | abs(GPU/WRF) | WRF negative visits | GPU negative visits |
|---|---:|---:|---:|---:|---:|
| qc | -6.1276037e-06 | -2.7068268e-06 | 0.441743 | 656,722 | 559,774 |
| qr | -2.9188398e-06 | -2.8539434e-06 | 0.977766 | 435,141 | 434,606 |
| qi | -3.4104044e-06 | -1.9839547e-06 | 0.581736 | 2,287,995 | 2,248,197 |
| qs | -8.9028536e-06 | -8.9406567e-06 | 1.004246 | 4,457,977 | 6,299,023 |
| qg | -3.3292793e-06 | -3.2550004e-06 | 0.977689 | 3,561,370 | 5,941,615 |
| Nr | -425.31412 | -495.44864 | 1.164901 | 504,272 | 499,665 |
| Nc | -64750280 | -64036036 | 0.988969 | 843,352 | 509,618 |
| Nn | -2.7093901e+08 | -1.2053447e+08 | 0.444877 | 303,279 | 265,687 |

Observed original-WRF wall: 1047.047 s, concurrent nice-19 CPU run including initialization and diagnostic I/O. This is not an S1 benchmark or an observer-overhead measurement.

Full CPU all/ring/interior quantiles and WRF values at the exact GPU-minimum step/cell are in the accompanying JSON. GPU quantiles were not captured by v-run and are unavailable; no quantile-equivalence claim is made.

## Accepted narrow disposition; residuals remain OPEN

The existence, scale and predominantly interior location of these intermediate negatives are reproduced by the original model, with an observer proven output-byte-neutral. This supports a narrowly worded known issue about WRF intermediate states, not a general clipping fix.

**Residual measured excesses are explicit:** default QI has abs(GPU/WRF)=1.140410 and QG=1.394381; WDM QS=1.004246 and Nr=1.164901. WDM snow/graupel visit counts also differ substantially. The compared forecasts evolve independently; these ratios alone do not identify an erroneous operator. No frozen cross-trajectory extremum-ratio acceptance band is specified in this lane contract, and none is invented here.

The strict interpretation "every GPU amplitude must lie within the pristine trajectory envelope" is therefore not green. Same-input operator replay for the residual fields would require their actual GPU RK/source/flux/mass operands. The retained current GPU receipt contains counts/extrema but no such operand packets; the current optional packet captures QC only. No GPU replay or additional instrumented run was authorized/executed by this CPU-only lane.

Manager decision (2026-10-14, direct task message): accept the narrow known-issue outcome; leave residual QI/QG/Nr/QS extrema and WDM QS/QG visit-count differences OPEN and disclosed. No targeted GPU capture is requested. This is a proof-only handoff to s2-integrate, not global F2 clearance. No scientific tolerance, guard or source change is proposed.

## Suggested known-issue wording

> In matched 24-hour Swiss Thompson and WDM6 runs, original WRF 4.7.1 itself produces millions of predominantly interior negative moisture/number values between the final RK transport step and microphysics, including cloud water around −6×10⁻⁶ kg/kg and number concentrations of order 10⁸/kg. The GPU has the same magnitude classes. This is not purely roundoff and does not justify every GPU event: selected GPU extrema exceed those of the independently evolved original-WRF trajectory. In WDM6, GPU snow/graupel negative visits are 6,299,023/5,941,615 versus WRF 4,457,977/3,561,370. Those visit-count differences and the residual QI/QG/Nr/QS extrema remain OPEN; exact per-event operator attribution is not established. No non-WRF clipping has been added.

## Provenance, resources and scope

See README.md for the seam, original-WRF source mechanism, source/archive identity checks, reader controls and precise limitations. The original physics archive retains 403 objects byte-identical; only the observer-bearing solve caller is rebuilt and one observer object is added. Product source, launches, allocations, synchronization and compile path are unchanged. No GPU or performance claim is made. Raw original-WRF event streams, field hashes and actual-run receipts are retained under `<USER_HOME>/wrf_gpu2_lanes/t-f2neg/` and bound by the evidence manifest.
