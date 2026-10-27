# Stochastic physics and input perturbations

**Status: Experimental.** In v0.4.0 the stochastic physics options (`&stoch`) ran but had not been compared with the original CPU WRF over 24 hours. Since then, single-domain 24-hour runs with SKEBS, SPPT and SPP match original CPU WRF runs with the same namelist and seeds (Swiss case, [GPU check](#gpu-qualification-swiss-d01-24-h)); nested runs have not been checked yet. `gpuwrf perturb` runs on the CPU, offline.

`gpuwrf perturb` prepares offline ensemble member inputs. It currently supports
**spatially uniform fractional wind perturbations** of U and V. Temperature,
moisture, pressure and spatially varying balanced perturbations are not implemented.

```bash
gpuwrf perturb --case inputs/case --output inputs/wind-members \
  --members 5 --seed 451 --wind-std 0.05
gpuwrf ens prepare --mode batch --cases inputs/wind-members/member-* \
  --hours 24 --output-dir ensembles/wind
```

The command creates new member case directories containing private copies of
`wrfinput_dNN` and `wrfbdy_dNN`. Fixed `namelist.input` and `wrflowinp_dNN` inputs
are linked to the source case; retain the source case while using the members.
It refuses overwrites, missing/packed/nonfinite selected fields and incomplete
boundary wind records. `--ic-only` explicitly leaves lateral boundaries unchanged
and permits a case without `wrfbdy_d01`. Source inputs are never edited.

For N members the recipe draws N standard normal values using NumPy PCG64 with
the unsigned 64-bit seed. It subtracts their ensemble mean, then rescales their
sample standard deviation (N−1 denominator) to `--wind-std`. A member's multiplier
is one plus its rescaled value. The example uses a **dimensionless 5% sample
spread**. Before REAL32 rounding, the finite ensemble has mean multiplier one;
normalization makes the members dependent. A recipe producing nonpositive multipliers is refused.
The manifest stores the full-precision factors and the REAL32 factors actually
used, so exact recipes can be inspected and replayed independently of RNG version.

One multiplier scales U and V throughout every nest, level and boundary record
of a member. Both boundary values (`U_BXS`, etc.) and time tendencies
(`U_BTXS`, etc., including the final record) receive the same multiplier. WRF's
mass/map-factor coupling is linear in wind with mass fixed, so this operation
commutes with coupling and with time differentiation, up to REAL32 rounding.
Wind units remain m/s and coupled boundary units remain their original WRF units.
All other fields, attributes, dimensions and file formats are retained.

This is an **unbalanced input recipe**: it preserves wind direction and the
coherent fractional IC/BC change, but provides no geostrophic balance, calibrated
ensemble spread, forecast skill guarantee or spatial covariance model. It does
not use WRF's RNG and does not enable stochastic physics during a forecast.

`perturb_manifest.json` records source/prepared SHA256 hashes, exact multipliers,
seed, sampling definition and changed-cell counts. Prepared cases feed the existing
`ens prepare batch` scheduler; this command itself never runs a forecast or uses
JAX/GPU. Storage grows by the copied IC/BC files per member. Peak array workspace
is proportional to the largest selected wind variable; there is no stacked
member array. Hashes detect input edits during preparation.

## Opt-in WRF stochastic physics

The default is OFF. Select WRF controls in `&stoch`, for example:

```fortran
&stoch
 skebs = 1,
 sppt = 1,
 spp_pbl = 1,
 rand_perturb = 1,
 nens = 7,
/
```

`skebs=1` (also the original `stoch_force_opt=1` alias) generates wind and
potential-temperature backscatter tendencies. `sppt=1` multiplies the combined
WRF physics tendencies for U, V, potential temperature and water vapor, using
the capped dimensionless pattern. Other moisture species are unchanged.
`rand_perturb=1` generates and exports WRF's random diagnostic field; the
upstream driver applies no separate model tendency from it.

`spp=1` enables all three SPP selectors. Independent `spp_conv`, `spp_pbl` and
`spp_lsm` arrays are also accepted, including child-domain selections. WRF
allocates all three fields when any SPP selector is enabled; the selectors
separately control generation and consumption. Scalar Fortran array assignments
set the first domain entry; write vectors for independent domain selections.
The original `spp` master overrides all three selector arrays globally.

MYNN SPP changes momentum/heat diffusivities, the cloud-PDF humidity numerator,
plume entrainment before its limiter, and MYNN surface-layer local momentum and
thermal roughness. It consumes the previous held pattern, then the first RK
part2 updates patterns for the current tendencies and child forcing. Perturbed
roughness is local, so it does not repeatedly multiply the stored surface ZNT.
WRF 4.7.1's cumulus driver explicitly resets its local SPP controls to zero;
RUC copies the LSM pattern but does not use it in the soil arithmetic. These two
selectors therefore generate diagnostic patterns with the same upstream physical
no-op. No generic tendency multiplier is substituted for either parameterization.

The native profile requires physics and WRF RK source coupling. SPPT currently
requires PBL 0/5 and cumulus 0/1; GWDO requires MYNN. SPP with GSL GWDO=3 is
refused. Values other than 0/1, unknown controls, nonfinite parameters and
unsupported multi-perturbation/boundary extensions fail closed before execution.
Registry INTEGER controls require signed INTEGER4 values, including vertical
modes and spectral bounds; fractions, quoted numbers and logical values are
refused. Custom `kminforct`, `lminforct`, `kmaxforct`, `lmaxforct` bounds are
SKEBS-only. WRF requires their defaults whenever SPPT, RAND or any SPP selector
is active in any domain, including an independent child selector.
`hrrr_cycling=.true.` is required for the restart profile. Fourier setup needs
at least 12 points on each full horizontal stochastic extent. All source defaults
come from `Registry/registry.stoch`; the LSM length scale is 50 km. Small domains
may need explicitly smaller Gaussian length scales to resolve their spectrum.

## Reproducibility and qualification

The stochastic physics is checked on the CPU against the original WRF code and,
for 24-hour single-domain forecasts, on the GPU against original CPU WRF runs with
the same namelist and seeds (see [GPU check](#gpu-qualification-swiss-d01-24-h)).
Nested runs, including the Tenerife three-nest cases, have not had that GPU check yet.
The RNG profile matches GNU Fortran 14.3's eight-word `RANDOM_SEED` state and
xoshiro256** REAL4 stream, including original WRF signed INTEGER4 overflow,
seed/member/date mixing and Gaussian rejection. WRF RNG is compiler-defined;
another compiler's stream is not promised. Seeds `iseed_skebs`, `iseed_sppt`,
`iseed_rand_pert`, `iseed_spp_conv`, `iseed_spp_pbl`, `iseed_spp_lsm`, `nens`
and the start date determine the stream. There is no host RNG in stepping.

The full carry stores REAL32 spectral coefficients, held fields, normalization
plans, INTEGER4 seeds and pre-cap event counters. Integer64 operations implement
the RNG recurrence; no floating-point64 stochastic carry is introduced. The
FFT follows original unnormalized FFTPACK5 ordering. The vertical imaginary
assignment deliberately reads the already-updated real coefficient, as WRF does.

Original CPU-WRF three-step histories and restart seeds check all eight fields
with vertical structure off/on. Seven independent 1,000-step original-WRF
trajectories check amplitude and spectra, including 5,600 exact seed words.
MYNN signed land/water/snow surface, active plume-limiter, real drizzle-cloud and
diffusivity-height regimes have separate checks against the original source. SKEBS/SPPT
C-grid tendency adapters use the original source expressions and split mass,
including WRF's initialized high-side mass ghost. These component checks and
short CPU coupled runs do not establish full forecast drift or station skill;
the 24 h GPU runs against CPU-WRF are described below.

Children receive Registry-staggered parent SINT fields through the existing
nest forcing path. They hold no Fourier plans and advance no independent RNG.
Both NetCDF and pickle restarts preserve every stochastic leaf and the next
step exactly. Active checkpoints missing the carry fail closed; old OFF
checkpoints remain usable. Active history adds eight native pattern fields and seven `STOCH_CLIP_*`
pre-cap counters, and fills the six native seed arrays from the live streams.
The existing OFF output schema is retained.

With stochastic physics off, there is no extra device state or step work; the
default forecast programs are checked to be unchanged with
`proofs/s1_port/release_identity.py`.
With it on, the model uses FFTs, fused pointwise forcing and an exact parallel
form of WRF's serial random stream. Every polar Gaussian proposal consumes
exactly two uniform numbers, so parallel lanes can start at known positions of
the stream (GF(2) jump-ahead of the xoshiro256** state, with polynomials
precomputed on the host), draw their proposals with the same arithmetic as the
serial kernel, and keep the first accepted values in order. The next seed is the
state after the last proposal used. The result is bit for bit the same as the
single-thread serial kernel on CPU and GPU (random values and seed words); an
exact serial fallback covers the case of too few proposals, which has never been
observed. For one model step (seven chained pattern draws) on an RTX 5090 this
takes 0.36 ms instead of 4.85 ms on the 43x43 Swiss grid and 0.49 ms instead of
21.4 ms on the 71x121 Canary Islands 9 km grid. There are no per-draw launches and no
host round trips per time step. Vertical-constant fields are stored with one
level; child domains keep only fields.

Reproduce the component checks with `tests/test_stochastic_*.py`. Original arithmetic
fixtures and compiler/source hashes are tracked in `tests/fixtures/u2_stoch_real4.*`;
build recipes and the actual CPU-WRF reference staging are in `proofs/u2_stoch/`.
Temporary originals are copied privately, preserving executable `.text` bytes.

## Stochastic ensembles

`gpuwrf ens prepare --mode stochastic` writes N member namelists from one case
with SKEBS and SPPT (optionally SPP), identical `iseed_*` and `nens` = 1..N,
following WRF's ensemble convention, and records each member's seed words.
`--mode multiphysics` uses a table of physics combinations that passed the
v0.4.0 GPU-vs-CPU-WRF D6 gate, optionally with SKEBS on top. Both refuse
combinations the model or WRF would not perturb (SPPT outside PBL 0/5 and
cumulus 0/1, SPP without MYNN) and Gaussian length scales that the model's
spectral setup cannot resolve on the case grid. See
[ensemble.md](ensemble.md#stochastic-and-multiphysics-presets) for usage, the
physics table and the Berner et al. (2011) comparison of the two approaches.
A plan for that comparison on the Tenerife three-nest cases is in
`proofs/w41_stoch/BERNER_PLAN_V050.md`.

## GPU qualification (Swiss d01, 24 h)

Each arm runs the default Swiss case (Thompson, MYNN, Noah-MP, RRTMG, no
cumulus; 43x43x45 at 3 km, dt 18 s) for 24 h on an RTX 5090 with only an
`&stoch` group added, against an original serial CPU-WRF 4.7.1 run of the
**identical namelist and seeds**. Gates: the frozen D6 manifest over every
paired hourly frame (`proofs/v040/final_validation/score.py`: all original
history fields present on GPU, raw finite audit of both arms), every stochastic
history field and seed array against CPU-WRF (`proofs/w41_stoch/pattern_compare.py`,
pre-registered bound `8e-6 + 64 eps` of the field maximum, seeds exact) and the
compiled-kernel resource check. Gaussian length scales are 15 km (see the length-scale note).

| run | `&stoch` | D6 vs CPU-WRF (fields x hourly frames) | stochastic fields vs CPU-WRF (worst error / field max), seeds | kernel check | seconds per forecast hour |
|---|---|---|---|---|---|
| OFF (reference run of the same case, 2026-10-16) | none | PASS, 362 x 25 | none written | PASS | 0.91 |
| SKEBS | `skebs=1` | PASS, 365 x 25 | 3 fields, 5.2e-7; seeds exact | PASS | 1.00 |
| SPPT | `sppt=1` | PASS, 363 x 25 | 1 field, 6.2e-7; seeds exact | PASS | 0.98 |
| SPP | `spp=1` (convection, PBL, land) | PASS, 374 x 25 | 12 fields, 1.9e-6; seeds exact | PASS | 1.01 |
| all | SKEBS + SPPT + SPP + `rand_perturb`, vertical structure on, `nens=7` | PASS, 379 x 25 | 17 fields, 1.6e-6; seeds exact | PASS | 1.12 |

The field counts are the full set of history fields compared in every frame.
The frozen D6 limits apply to 62 of them (temperatures, winds, pressure,
moisture, precipitation and similar); the others are reported without a
limit, and the stochastic fields and seeds are checked separately (column 4).
Every field that CPU-WRF writes is present in every GPU frame and no frame
holds a NaN or infinity. The only guard repairs are the hydrometeor
non-negativity repairs that the OFF run also makes, with similar counts
(172,848-176,737 against 174,717 over 24 h); they belong to the default
program and remain an open item (negative moisture visits). The
GPU files carry seven extra `STOCH_CLIP_*` counters (port diagnostics, not
written by WRF). Before this qualification the GPU history lacked SPP's
`FIELD_CONV`, `FIELD_PBL` and `FIELD_SF`, which CPU-WRF writes as zero when
SPP is on without Grell cumulus; they are now written. With all schemes on, the
earlier single-thread random stream took 1.99 s per forecast hour; the
parallel stream gives 1.12 s with identical patterns and seeds.

Times are the median of hours 2-24 of cold single-process runs on a shared
machine, not benchmarks; the OFF run is from another session and source tree,
so the differences are indicative only. Records: `proofs/w41_stoch/GPU_QUALIFICATION.md`.
