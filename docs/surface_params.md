# Surface parameters as traced inputs (Track P) — experimental

`gpuwrf run --surface-controls CONTROLS.npz` replaces selected Noah-MP surface parameters with per-cell value
fields **without recompiling**: the fields are runtime inputs of the compiled step program, so every parameter
set of one grid reuses the same cached executable (same AOT key). Without the option the released program is
unchanged (frozen release identities Swiss `3356e249…`, PROD d01 `196aed9f…` verified on this feature's tree).

Status: experimental, opt-in, Noah-MP (`sf_surface_physics = 4`) only. Checked so far:

- every control equals running the released code on an MPTABLE edited the same way (byte-exact, every control);
- two different parameter sets use one compiled program (same cache key on CPU and on the GPU: a second Swiss
  24 h run with other values loaded the first run's executable and compiled nothing);
- Swiss 24 h on the GPU with unchanged (prior) values passes the frozen field tolerances against original
  CPU-WRF (T2 RMSE 0.085 K, 10 m wind 0.04 m/s);
- forecast-level check against original CPU-WRF: a Swiss 24 h run with eight edited parameters moves the GPU
  forecast like the same MPTABLE edit moves CPU-WRF, compared hour by hour over land. LAI responds identically;
  median pattern correlations are 0.97–1.00 for emissivity, albedo, 10 m wind and friction velocity, 0.85–0.90 for
  sensible/latent heat flux and 0.49–0.59 for skin and 2 m temperature. The median hourly response amplitude
  (GPU RMS / CPU RMS) is 0.986–1.009 for every field, but single hours vary more (2 m temperature 0.92–1.07, skin
  temperature 0.89–1.10, latent heat 0.93–1.13). The leftover GPU–CPU response difference has about the same size
  as the ordinary difference between a GPU and a CPU-WRF forecast of the unedited case, which is consistent with
  it being run-to-run noise, but this is an inference, not a proof of its origin;
- resuming a run with different, added or missing controls is refused.

Not done in v0.4.1: calibration against real stations. The EKI tools below are a scaffold, checked on a synthetic
twin experiment (OSSE) only; no parameter set has been estimated from real observations yet. The guarded
real-station runs and the roughness pilot are planned in `proofs/w41_trackp/P1_GPU_PILOT_PLAN.md`.

## What can be controlled

| Control | Replaces (Noah-MP consumer) | WRF source | Needs |
| --- | --- | --- | --- |
| `z0mvt` | vegetation roughness → Z0M/Z0WRF → ZNT → surface layer | MPTABLE `Z0MVT(VEGTYP)` | – |
| `eg` | ground emissivity → EMG → FIRA/TRAD/EMISS | MPTABLE `EG` (soil) | – |
| `albsat`, `albdry` | saturated/dry soil albedo (vis, nir) → GROUNDALB → SAV/SAG, ALBEDO | MPTABLE `ALBSAT/ALBDRY(SOILCOLOR)` | – |
| `rhol`, `rhos` | leaf/stem reflectance (vis, nir) → two-stream | MPTABLE `RHOL/RHOS(VEGTYP)` | ρ + τ ≤ 1 |
| `laim`, `saim` | monthly LAI/SAI (12 months) → phenology → ELAI/ESAI | MPTABLE `LAIM/SAIM(VEGTYP,month)` | `dveg` 1, 3 or 4 |
| `shdmax` | annual-max green fraction, FVEG = SHDMAX | wrfinput `VEGMAX/100` | `dveg = 4` |
| `vcmx25`, `mp` | Ball–Berry photosynthesis / stomatal slope → transpiration | MPTABLE `VCMX25`, `MP` | `opt_crs = 1` |
| `ctopo` (host control) | topo_wind drag factor → YSU/Shin-Hong momentum surface drag | start_em `CTOPO` | `topo_wind` 1/2, `bl_pbl_physics` 1/11, no two-way feedback onto the domain |

`ctopo` is a *host* control: CTOPO is already a runtime input of the step program, so the control replaces its
value when the run starts and adds nothing to the program — a file with only `ctopo` runs exactly the default
program of the case (same lowered program text for any value, checked on the CPU). It takes a domain-wide or
per-cell factor f of the correction, CTOPO' = 1 + f (CTOPO − 1) rounded once to REAL, so cells without a correction
keep CTOPO = 1 and f = 1 reproduces the start_em field byte for byte; negative results are refused. Not supported
in v0.4.1 (refused): two-way feedback onto a topo_wind domain, B > 1 lanes, checkpoint/resume (the restart digest
covers the traced controls only).

Each field has the shape the consumer reads ((ny, nx), (2, ny, nx) for vis/nir, (12, ny, nx) for months) and
the consumer's dtype. A control whose consumer is not active under the chosen Noah-MP options is refused, as are
non-finite values, values outside physical bounds and unknown names. Not controllable (by design, documented):
soil hydraulic parameters (Noah-MP WATER reads its own soil table), canopy height `HVT` (two consumers), Jarvis
`RSMIN` (`opt_crs = 2`), snow parameters, land-use category and terrain (geometry/discrete physics).

Semantics: a field built from multipliers is `round_once(table value × multiplier)` per cell — for a per-category
multiplier this is byte-identical to running the released code on an MPTABLE whose row was edited the same way
(tested for every control). Glacier and lake cells do not run Noah-MP SFLX, so controls there are inert.

## Workflow

```bash
# 1. once per case: the float64 table basis of every control (CPU is fine)
JAX_PLATFORMS=cpu gpuwrf surface-params basis --input-dir CASE --output basis.npz
# 2. any number of parameter sets (NumPy only, seconds); entries without a prefix are d01
gpuwrf surface-params expand --basis basis.npz --output rough.npz \
    --class z0mvt:2=1.25 --class z0mvt:5=1.25 --set albdry=0.95
# 3. run (no recompile across parameter sets of one grid); without domain options run is d01 only
gpuwrf run --input-dir CASE --output-dir OUT --surface-controls rough.npz
```

Per-domain entries are optional (`d01`, `d02`, …); domains without an entry run the released program. For a
nested case, prefix entries with their domain (`--set d02:albdry=0.95`) and run all domains with
`--domains-from-namelist` (or `--max-dom N`): a control for a domain outside the run is refused.
`basis` refuses ALISIOS data before reading anything: the case directory (directory links followed, files named in
the namelist), and every runtime root the load reads (WRF tables via `GPUWRF_WRF_ROOT`, asset and compile caches,
paths named in the environment) must not resolve into an ALISIOS data root, and any other open of such a path
during the load is refused. Copy such inputs once under `alisios_guard` and pass the copy.
The control file carries a canonical manifest and per-domain content digests; altered files are refused.

## Restart

The content digest of each domain's controls is part of the nested restart identity (and of the single-domain
checkpoint payload). Resuming with different, missing or added controls is refused.

## Batched members (B > 1)

The batched-ensemble path (`GPUWRF_BATCH_ENSEMBLE=B`, one vmapped program for B lanes of one grid) accepts one
control file per lane (`NestedPipelineConfig.batch_surface_controls`); every lane must carry the same control
names (one program signature). Lanes may share an input directory (members of one case) or use different dates.

## Ensemble Kalman inversion scaffold (synthetic tests only in v0.4.1)

`gpuwrf surface-params eki init|update` manages a small parameter ensemble in log-multiplier space
(Tikhonov-regularised EKI, host float64, tempering by `--dt`), writing one control file per member. Members are
forecasts with `--surface-controls member_NNN.controls.npz`. `eki collect --member-dir … --stations S.csv --obs
O.csv --sigma t2=1.0 --sigma wind=1.5` samples every member's wrfouts at the stations with the `gpuwrf verify`
operator (one set of extraction options, earth-relative winds by default) and keeps the station/time/variable keys
that all members and the observations share, in one fixed order → `predictions.npy` (members × obs) and `obs.npz`
(`y`, `sigma`) for `eki update`. Inputs whose real path (symlinks followed) lies under an ALISIOS data root —
observations, station metadata or member wrfouts — are read only inside ONE `alisios_guard` session that admits all
of them and stays open until the last member is read (`--alisios-issue-days/--alisios-valid-days`); without the guard
such inputs are refused before anything is opened. Purely local inputs need no guard.
Parameter spec:

```json
{"parameters": [
  {"name": "z0:forest", "control": "z0mvt", "classes": [1, 2, 3, 4, 5], "prior_sd_log": 0.4, "bounds": [0.3, 3.0]},
  {"name": "z0:shrub",  "control": "z0mvt", "classes": [6, 7, 8, 9, 10], "prior_sd_log": 0.4}],
 "tikhonov": true}
```

Status: the update and the station operator are tested on synthetic data and a twin experiment; a run against
real ALISIOS stations (guarded, ALLOW days only, observations before 2026-07-01) is planned, not done.
Limits: updates live in the span of the initial ensemble (≤ members − 1 directions), spread is not a posterior,
repeated hourly observations are correlated (temper or inflate `sigma`). Tune only what stations constrain;
hold out dates and station blocks before believing any improvement (see the design critique,
`.agent/decisions/evidence/DESIGN-SURFACE-PARAM-ESTIMATION-CRITIQUE-astra.md`).
`gpuwrf surface-params osse` runs an offline Noah-MP twin experiment (synthetic truth, CPU, minutes) that
recovers a two-group roughness multiplier with one compiled, member-vmapped land program.
`gpuwrf optimize` runs the whole loop with real forecast members (members, iterations, resume, held-out prior vs
posterior, one program for every member): see
[Continuous parameters](regional_optimization.md#continuous-parameters-experimental).

Proposed support-matrix row (v0.4.1): *Track P surface controls — experimental, opt-in; table-edit oracle
(byte-exact), one compiled program per grid (CPU + GPU), Swiss 24 h GPU forecast response matches original CPU-WRF
on the edited MPTABLE (median hourly response ratios 0.986–1.009); restart refuses mismatched controls. EKI calibration: scaffold + synthetic twin (OSSE)
only — not done against real stations in v0.4.1.*
