# Ensemble design for `gpuwrf da` (experimental, v0.4.1)

This page covers the ensemble-side options of the offline LETKF (`gpuwrf da analyse|cycle`, see
[da.md](da.md)): how the prior is inflated, how spread is kept in the written members, how observations are
localised between variables, and how initial and boundary perturbations are built. Everything here is **opt-in**:
a configuration without an `inflation` section runs exactly the probe behaviour of v0.4.0-dev.

Status: experimental. The final design passed a prospectively registered Lorenz-96 experiment on untouched
seeds. In real-model CPU-WRF twin experiments it kept its spread over three cycles and lowered the analysis
error, but it was overdispersive and failed the calibration check, both in the first run and in the prospectively
registered run of the final code (spread 2.3–2.6 times the error). It has not been qualified on the GPU member
engine or against real observations. See *Evidence*.

## Why these options exist

A development study on three Canary Islands cases (not in this repository) found that the first
ensemble design lost 40–75 % of its temperature and wind spread over each 6-h forecast, that the surface
innovations were dominated by a systematic observation-minus-model offset (T2 about −1.9 K, 10-m winds
+1.3…3.2 m/s) which a single global inflation factor mistook for missing spread, that RTPS 0.8 relaxed the
analysis back toward the *raw* prior so the inflation never reached the written members, that co-located surface
winds degraded the analysed T2 through sampling noise, and that one shared boundary file left the outer rows
almost unperturbed.

## The `inflation` configuration section

```json
"inflation": {
  "method": "adaptive",
  "adaptive":    {"state_in": "/runs/da/cycle_02/adaptive_state.json", "state_out": "/runs/da/cycle_03/adaptive_state.json",
                  "rho0": 1.0, "var0": 1.0, "q": 0.04, "rho_min": 1.0, "rho_max": 10.0, "n_min": 2},
  "obs_bias":    {"by": "group", "apply_kinds": ["surface"], "correct": true, "var0": 0.25, "q": 0.01,
                  "state_in": "/runs/da/cycle_02/obs_bias.json", "state_out": "/runs/da/cycle_03/obs_bias.json"},
  "maintenance": {"method": "rtps", "alpha": 0.3},
  "additive":    {"pool_npz": "/runs/da/pools/additive_12z.npz", "amplitude": 0.1, "seed": 3}
}
```

File paths in this section must be absolute: unlike `member_files`, they are not resolved against the config
file's directory, and a relative path is refused rather than read from the current working directory.
`inflation.chain_section(section, previous_dir, cycle_dir)` writes the per-cycle paths for a cycle loop.

- **`method: adaptive`** — one multiplicative inflation per state variable (T, QVAPOR, U, V) and height band
  (0–1, 1–2, 2–5, > 5 km AGL). Each cycle, every observation group in a band gives an estimate
  ρ_o = (Σd′²/m − σo²)/σb² from bias-removed innovations d′ and the raw prior spread σb (m = n − 1 when the
  group's own mean is removed), with sampling variance v_o = 2/n_eff·((ρσb² + σo²)/σb²)², n_eff = m scaled to the
  number of distinct profiles (station|site|launch) in the band — the levels of one radiosonde are not independent
  samples. It is fused with the previous cycle's value
  (Miyoshi 2011): sparse or low-signal groups barely move the estimate, so no hard clamp is needed. The state
  persists between cycles through `state_in`/`state_out` (random-walk variance `q` per cycle). The inflation is
  applied to the member deviations in state space (moisture in log space, positive by construction; the
  specified boundary zone keeps its bytes), so the transform runs with ρ = 1 and **RTPS relaxes toward the
  inflated prior**.
- **`obs_bias`** — an online scalar Kalman estimate of the mean observation-minus-background offset per
  observation group (`by: group`) or per station (`by: station`). With `correct: true` the values of the listed
  kinds are corrected before the analysis. The update gain uses the variance of the cycle mean innovation
  INCLUDING the ensemble's estimate of the forecast error that is coherent over the group (variance over members
  of the group-mean H(x)); without it a domain-wide flow error of one cycle is mistaken for a bias (Swiss twin
  OSSE). `var0` (prior variance) and `q` (per-cycle random walk) are in units of the group's σo². State files are
  schema `w41.obs_bias.v2`. This treats the offset as an observation/representativeness bias; it does not correct
  a model bias.
- **`maintenance`** — `rtps` (alpha, optional `target: inflated` with a fixed `rho`), `rtpp`,
  `multiplicative` (posterior factor) or `none`.
- **`additive`** — seeded, centred additive inflation after the relaxation: every member receives
  amplitude × a Gaussian combination of the pool's centred deviations (one weight matrix for all fields, so the
  multivariate structure is kept; QVAPOR/Q2 multiplicatively). Pools are written with
  `spread.write_pool_npz` (archive deviations, forecast differences or spun-up member deviations).

## Variable localisation

`varloc.as_var_loc(table, names)` produces the `localisation.variables` section:

- `surface_split`: sonde observations update all variables; T2/Q2 update only the thermodynamic group
  (T, QVAPOR, T2, Q2, PSFC); U10/V10 only the wind group (U, V, U10, V10).
- `strict`: every observation updates only its own group.

One set of LETKF weights is computed per group (about twice the weight cost of the plain analysis).

## Perturbations and boundaries

- `perturb.combination_perturbations` — member deviations as seeded Gaussian combinations of centred pool
  states, **including the column mass MU**.
- `perturb.rebalance_with_mu` — hydrostatic rebalance for T/QVAPOR **and** MU increments, using real.exe's
  top-down pressure recursion and the HYPSOMETRIC_OPT 2 geopotential (checked against the port's real.exe
  `balance` oracle). The fixed-MU rebalance of the probe refuses MU increments.
- `perturb.forecast_deviations` — deviations of forecasts valid at the same time (spun-up, lagged or NMC
  pairs), which are model-balanced.
- `lbc_ens.write_member_wrfbdys` — one perturbed `wrfbdy_d01` per member whose first record equals the
  member's initial deviation (the IC is then written untapered for U/V/T/QVAPOR; MU stays tapered because the
  boundary file keeps MU frozen) and whose later records follow the member's perturbation in time.

## Evidence

See `proofs/a1_da/ens/ENS_DESIGN.md` for the Lorenz-96 OSSE, the real-model CPU-WRF twin experiment on the
Swiss example case, the balance experiment and the cost numbers.

In short:

- Lorenz-96 (40 variables, N = 20, model error and an observation bias): analysis RMSE 0.752 with the old design,
  0.610 with this one; spread matches the error (ratio 1.03).
- Swiss twin experiment (original CPU-WRF members, N = 20, three 3-h cycles): the old design's spread collapsed
  to about 2 % of the error and its inflation factor escalated (182 → 670 → 1653). The new design kept its spread
  in every cycle, and its analysis error was 39 % lower.
- The new design was **overdispersive**: spread was 1.6–2.1 times the actual error in the first run and 2.3–2.6
  times in the prospectively registered run of the final code on a new truth (registered calibration band 0.5–2.0:
  FAIL both times). The other registered checks passed (no spread decay, written members carry the analysis
  spread, analysis error 0.81 vs prior 1.31). The excess was already there before the first analysis, so the size
  of the initial and boundary perturbations is the main suspect. Inflation never goes below 1: if your ensemble
  already has more spread than error, reduce the perturbation amplitude, `maintenance.alpha` and
  `additive.amplitude` rather than relying on the adaptive inflation to remove spread.
- Balancing the perturbation's column mass (MU) made no measurable difference to how much spread survives the
  first forecast hour in that small domain; the per-member perturbed boundary files are what kept the spread.
- `obs_bias` needs many cycles. A constant station bias and a forecast error shared by the whole domain cannot
  be told apart within a couple of cycles.
