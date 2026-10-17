# Observation nudging: pristine WRF proof packet

This packet qualifies the root-domain `obs_nudge_opt=1` chain on CPU: processed
OBS_DOMAIN decoding, normalization/QC/projection, multi-level errors and
spreading, resident windows, actual physics/RK source coupling, both pipeline
bindings, admission and held-state restart. MYNN PBL/SFCLAY are required.
GPU and long coupled forecast qualification remain pending.

The earlier single-level partial seal is retained as history; the full chain
adds the evidence below without treating a previous port version as truth.

* `full/savepoints/ERROR_MANIFEST.json`: 8 pristine REAL4/REAL8 flag cases,
  three `errob` calls each, including surface, pressure/height location and
  gross-temperature rejection with persistent VAROBS/RKO.
* `full/savepoints/SPREAD_MANIFEST.json`: 40 pristine grouped-sounding and
  negative-QV cases, six calls each; all four variables, both surface schemes,
  PBL exclusion, missing first level and maximum sounding gaps.
* `full/savepoints/PROJECTION_MANIFEST.json`: 36 original full `module_llxy`
  and unchanged `rotate_vector` evaluations, REAL4/REAL8, northern/southern
  LC/PS/Mercator and tangent LC; bit-exact scalar host results.
* `full/savepoints/COUPLED_MANIFEST.json`: actual clean DM CPU-WRF physics/RK
  boundaries at three Swiss steps. The write-only observer left all 362
  variables in each of four history files byte-identical. The complete raw
  reader → driver → coupled source passes the predeclared REAL operand bounds.
* `full/savepoints/WINDOW_MANIFEST.json`: twelve original Swiss steps with
  rolling window shifts, one future report, surface overwrites, TEMP replacement,
  PILOT level-count preference and outside-domain controls.

The production CPU smoke executes three real Swiss steps against the original
DM WRF run: zero nonfinite values/guards and all checked frozen D6 core bounds
pass. After-step1 checkpoint/resumed hold-step2 matches all 183 carry leaves
byte-for-byte. A second three-step run with grid FDDA and station nudging together
also passes against pristine WRF with both features active. These are short CPU
qualification runs, not 24-hour forecast or GPU performance claims. P/PH remain
report-only under the frozen manifest.

The daily-loop critic fix is recorded in `full/receipts/daily/`. Six new
regressions exercise the actual host loop at hourly/15-minute boundaries,
ionf holds, full-carry checkpoint continuation, Noah-MP diagnostic routing and
expired observation sources. Their model arithmetic is a recording hook except
for the real observation-driver test on pristine Swiss operands. A mutation of
the production stepper's clock is killed by the second-hour step assertion.
On the source commits recorded in its receipts, `full/daily_reuse.py` proves
all 528 other source blobs unchanged from the critic-accepted candidate,
carrying forward its scientific and release evidence.

`full/receipts/output/` closes the subsequent absent-history consumer finding.
Four additional regressions use actual observation initialization, the daily
loop, both real output consumers and real hourly/auxhist NetCDF writers. They
verify bulk Q2 when history is absent and held Q2 when present, including retained
land/radiation inputs. Only timestep and diagnostic numerical work is stubbed.
`full/output_reuse.py` binds the final composition with main's initialization
ownership guard and optimizer changes, checking unchanged numerical model ASTs.

`assimilation/obs_nudging.py` ports pristine WRF's time weights, surface vertical
influence heights, dynamic-initialization ramp and single-level `nudob` branches.
The slab routine accepts WRF innovations, not observations or gridded analyses.
It handles U/V/theta/QV, both surface horizontal schemes, both surface vertical
schemes, PBL exclusion and single-level PBL separation, saved-weight cadence,
restart refresh, boundary ownership, mass/map coupling, and coefficient/switch
application. Arrays are `(k,i)` with WRF 1-based station and PBL coordinates.
The operational `obs_driver` wraps these primitives with a rolling mutable
frame and four SAVWT fields. Surface ratios/regimes come from the actual
surface producer; KPBL uses MYNN's actual geometry and PBLH output.

The oracle extracts **unchanged** `nudob`, `get_timewt`,
`calc_rcouple_scales` and `get_vif_hts_slab` from pristine WRF into a lane-local
build. Source checksums, compiler and flags are recorded in the manifest.
Unused module names are resolved by empty modules; no constants or arithmetic
come from those stubs. WRF message/fatal wrappers print/stop. `DM_PARALLEL` is
selected because it initializes `kpbl_obs` and `rko` from `ERRF(5/9)`; serial WRF
leaves the former uninitialized in `nudob`. There are no communication calls in
the extracted routine. The separate full caller probes above qualify the
integration that this isolated operator oracle cannot establish.

The 32 REAL4/REAL8 cases use original CPU-WRF Swiss h1 pressure, geopotential,
mass, map factors and hybrid coefficients with synthetic innovations. Six calls
cover recompute, hold, recompute on restart, and varying TFACI. Each active case
contains surface and upper-air reports plus missing, outside-radius, and
outside-window controls. Tolerances are fixed operand-based arithmetic bounds
(manifest), not fitted to candidate errors. Helper fixtures separately exercise
time-window edges and the negative/-5000 PBL-relative VIF parameters.
Six additional REAL4/REAL8 cases bracket the fractional-level cutoff at
`NINT(rko+.45) == 1`; these use the same Swiss operands and unchanged oracle.

`io/obs_domain.py` decodes raw fixed-width records. A three-report specimen
(surface, single upper-air, two-level sounding) is checked against a Fortran
reader using the **original FORMAT statements** from `wrf_fddaobs_in.F`.
Decoding preserves raw QC/missing codes and sounding groups; it performs none
of WRF's subsequent QC, pressure/moisture conversion, projection or duplicate
replacement. See [the file-format guide](../../docs/OBSERVATION_NUDGING.md).

Reproduce with CPU environment and core/QUIET rules from AGENTS:

```sh
python proofs/s1_obsnudge/oracle/build.py <lane-build-dir>
python proofs/s1_obsnudge/oracle/make_savepoints.py <lane-build-dir> proofs/s1_obsnudge/savepoints
python proofs/s1_obsnudge/oracle/make_ingestion_savepoints.py <lane-build-dir> proofs/s1_obsnudge/savepoints
python proofs/s1_obsnudge/oracle/make_boundary_savepoints.py <lane-build-dir> proofs/s1_obsnudge/savepoints
python -m pytest tests/v040/test_s1_obsnudge.py --basetemp <lane-pytest-dir>
```

Full fixtures reproduce with `full/build.py`, `full/make_errors.py`,
`full/make_spread.py` and `full/make_projection.py`. `full/build_coupled_probe.py`
relinks a private diagnostic executable against read-only clean DM libraries;
`full/make_cpu_case.py`, `full/make_window_case.py` and `full/make_grid_case.py`
prepare private cases. No shared pristine build/run outputs are written.

The production source, optional carry and schema additions are selected by
`obs_nudge_opt=1`. The released default program is checked by the unchanged
o1 Swiss jaxpr probe. A station loop carries two grid accumulation arrays,
avoiding an `Nobs × grid` intermediate; mutable report storage is bounded by
the rolling frame. Immutable observation history is preloaded at initialization.
GPU launches, traffic, VRAM and cached readiness remain unmeasured. CPU driver
timings are component diagnostics, never GPU forecast speed evidence.
