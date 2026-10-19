# w42-icloud0 — `icloud=0` cloud–radiation decoupling switch (ALISIOS F7), v0.4.2

ALISIOS asked for one lever that lets a paired run separate cloud shading from flow-regime change: the same forecast
with radiation that does not see the clouds while microphysics and dynamics run unchanged. WRF already has it:
`icloud = 0`. The port now binds it (opt-in by namelist; the default program is unchanged).

## What WRF 4.7.1 does (pristine source audit)

* `module_radiation_driver.F:1309-1318` sets `CLDFRA = 0` on every radiation call; none of the `icloud` branches
  (`:1321-1508`) runs, so the `icloud_bl` MYNN merge is not used either.
* RRTMG LW/SW `IF (ICLOUD .ne. 0)` (`module_ra_rrtmg_lw.F:12032/:12180`, `module_ra_rrtmg_sw.F:10633/:10777`): no
  cloud water/ice/snow and no effective radii are read; the cloud inputs stay zero.
* `icloud` is read nowhere else in WRF (only the `ICLOUD` history attribute). Port: `physics_couplers` builds the
  same cloud-free RRTMG view (CLDFRA 0, condensates 0, no MP radii); `icloud_refusals` binds 0 with the same RRTMG 4/4,
  scheme-specific-view and cumulus restrictions as `icloud = 2/3`; history `CLDFRA = 0`.

## Evidence

| gate | result | receipt |
| --- | --- | --- |
| Z01 pristine `radiation_driver` (unchanged WRF library, generated caller) on 16 real WN3 0227 RE01 columns, `icloud` 0 and 1 | `icloud=0`: CLDFRA = CLDT = LWCF = SWCF = 0, GLW == LWDNBC bitwise, \|SWDOWN − SWDNBC\| ≤ 6e-5 W/m², clear-sky outputs bitwise equal between 0 and 1; clouds change SWDOWN by up to 736 W/m² | `fixtures/z01_radiation_driver_icloud.json` |
| Z02 port release path (`GPUWRF_FAST_DEFAULTS=1`, CPU interpret) vs Z01, frozen RE01 bounds | `icloud=0` worst 0.0012 of the bound (SWDOWN ≤ 0.005, SWDDIR ≤ 0.0002 W/m²), `icloud=1` control 0.002; port all-sky == clear-sky exactly | `fixtures/z02_port_release.json` |
| SWDDIR fix (#146) interplay | reverting the fix (w41-swddir mutant) leaves SWDDIR bit-identical under `icloud=0` (cloud-free beam), 190 W/m² off with clouds | Z02, `tests/v041/test_icloud0_oracle.py` |
| seam tests + mutants | real-call-path cloud-free view incl. BL clouds, MP radii and WSM3 restaging; refusal/catalog/step spec; 6/6 mutants killed | `tests/v041/test_icloud.py`, `fixtures/mutants.json` |
| GPU forecast (RTX 5090, tree `e0041902bd`) | WN3 0227 3-nest 24 h `icloud=0`, release defaults: rc 0, 75 wrfout, all finite, 21 GPU-min incl. cold compile; E41 PASS 4/4 (max STACK 32 B) | `gpu_arm.sh`, lane `gpu_arm/` |
| D6 vs original CPU-WRF 4.7.1 `icloud=0` (gen2 dmpar, 4 ranks 2×2, SUCCESS COMPLETE WRF) | PASS d01/d02/d03: 25/25 frames, 0 failures, 0 non-finite, coverage complete. Worst-lead RMSE d03: T2 0.070 K, PSFC 2.4 Pa, U10/V10 0.10/0.13 m/s, RAINNC 0.003 mm, W 0.031 m/s; d01/d02 T2 0.012/0.011 K | `fixtures/d6_summary_morning.json` |
| cloud-free radiation in the coupled run (report only) | history CLDFRA = 0 in both arms on every forecast frame (3 domains) while QCLOUD is present; SWDNB ≡ SWDNBC and GLW ≡ LWDNBC exactly in both arms; GPU−CPU worst RMSE SWDOWN 0.19, GLW 0.30, OLR 0.36 W/m² (d03). SWDDIR/SWCF/LWCF/CLDT are not in this history stream | `fixtures/icloud0_report_morning.json` |
| program identity of the merge candidate `45acb5652b` (+ w42-swdfid, w42-scorer) | location-stripped jaxpr and all closed-over constants of the d01 advance and the fused d02/d03 cascade IDENTICAL to `e0041902bd` (cheap keys differ: source/env fingerprint); the remaining main delta is host-side (history COSZEN = held radiation-call value). Control: the same tree with `GPUWRF_SWDDIR_V040_SPLIT=1` gives DIFFERENT jaxprs for both programs (tool sensitive to the swdfid code path) | `fixtures/PROGRAM_IDENTITY.json`, `fixtures/PROGRAM_IDENTITY_CONTROL.json` |

| release identities on the merge candidate `45acb5652b` | Swiss `b977712fe325…` / PROD `321aa651067f…` EXACT (default program unchanged) | `fixtures/release_identity.json` |
| CPU coupled smoke (Swiss RD11, real CLI loader, 3 steps) | `icloud=0` bound, own program (jaxpr sha differs from `icloud=1`), held CLDFRA 0 at step 1 and at the end, coupler view == held field 77,616/77,616, 0 non-finite (no clouds in these early Swiss steps: seam check only) | `fixtures/coupled_smoke.json` |

The final composed v0.4.2 candidate repeats the GPU arm once inside w42-bundle042's session: `BUNDLE_ARM_SPEC.md`.

## Tools

`driver_oracle.py` (Z01), `port_vs_driver.py` (Z02), `mutants.py`, `stage_inputs.py` (case from the w41-icloud2
guarded stage, no ALISIOS read), `cpu_ref.sh` (+ `../w41_icloud/cpu_reference.sh`), `gpu_arm.sh`, `gpu_trigger*.sh`,
`post.sh` (sequential D6 + `icloud0_report.py`), `d6_summary.py`, `program_identity.py`, `coupled_smoke.py`.
