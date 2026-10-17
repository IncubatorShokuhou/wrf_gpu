# Stochastic physics and input perturbations

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
unqualified multi-perturbation/boundary extensions fail closed before execution.
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

This is **CPU-oracle-qualified; GPU/coupled-forecast qualification pending**.
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

Original CPU-WRF three-step histories and restart seeds qualify all eight fields
with vertical structure off/on. Seven independent 1,000-step original-WRF
trajectories qualify amplitude and spectra, including 5,600 exact seed words.
MYNN signed land/water/snow surface, active plume-limiter, real drizzle-cloud and
diffusivity-height regimes have separate original-source gates. SKEBS/SPPT
C-grid tendency adapters use the original source expressions and split mass,
including WRF's initialized high-side mass ghost. These component gates and
short CPU coupled smoke do not establish full forecast drift, station skill or
GPU numerical/performance qualification.

Children receive Registry-staggered parent SINT fields through the existing
nest forcing path. They hold no Fourier plans and advance no independent RNG.
Both NetCDF and pickle restarts preserve every stochastic leaf and the next
step exactly. Active checkpoints missing the carry fail closed; old OFF
checkpoints remain usable. Active history adds eight native pattern fields and seven `STOCH_CLIP_*`
pre-cap counters, and fills the six native seed arrays from the live streams.
The existing OFF output schema is retained.

OFF introduces no device leaves or step work; full Swiss and PROD release
program identities are checked with `proofs/s1_port/release_identity.py`.
ON uses one serial-stream device kernel per produced innovation plane, FFTs and
fused pointwise forcing. There are no per-draw launches or timestep host round
trips. Vertical-constant fields are stored with one level; children retain only
fields. RNG serialization and FFT workspace are explicit performance risks;
actual GPU timings, peak memory and E41 qualification remain pending. CPU
compiler preflight lowers the RNG and both changed MYNN plume variants to
Triton GPU IR without floating-point64 operations; it does not execute a GPU.

Reproduce component gates with `tests/test_stochastic_*.py`. Original arithmetic
fixtures and compiler/source hashes are tracked in `tests/fixtures/u2_stoch_real4.*`;
build recipes and the actual CPU-WRF reference staging are in `proofs/u2_stoch/`.
Temporary originals are copied privately, preserving executable `.text` bytes.
