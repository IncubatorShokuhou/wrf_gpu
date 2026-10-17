# NTU multi-moment microphysics (mp_physics=56)

**Status in v0.4.0: Validated 24 h.** Passed a 24-hour comparison with the original CPU WRF in the v0.4.0 release check, in one tested single-domain configuration. Other combinations and nested domains are not validated. Its compiled GPU code exceeds the project's code-quality limits for speed and robustness; it runs correctly, but its speed has not been measured.

NTU is an opt-in microphysics option: set `mp_physics=56` in
`&physics`. `ccnty=1/2/3/4` selects the original WRF aerosol background; its
Registry default is 2. The checks use original WRF 4.7.1
`module_mp_ntu.F` and the `module_microphysics_driver.F:1586` argument map.

The port carries all seven moist fields and 21 scalar fields. Six scalars are
hydrometeor number concentrations. The other fifteen are dry/total/cloud/rain
condensation nuclei, potential ice nuclei, and ten ice shape/volume/moment
fields. These fields are transported with the appropriate moist/scalar
advection options, use available root boundary records, appear under the
original Registry history names, and survive lossless restart. Live nests
are refused; nested runs are not validated.

Cold initialization calls the original aerosol equations using the dycore's
dry-air specific volume, staggered geopotential and actual vapor. Restarts
retain the saved aerosol and moment state. History RAINNC and SNOWNC retain
the original REAL accumulation order. The land coupling retains original
public RAINNCV/SNOWNCV/GRAUPELNCV/HAILNCV rather than differences of rounded
totals. NTU's public RAINNCV is rain-only and SNOWNCV is snow-only, despite
the broader accumulated totals; this upstream partition is preserved and
disclosed. No effective radii or radar fields are fabricated.

Literal REAL4 and REAL8 checks cover 12 warm, cold, mixed-phase,
evaporating and ice-supersaturated columns, with all prognostics active across
the set and two evolving calls. All 8,568 public values per precision are
byte-identical to the original WRF code with zero-initialized scratch memory
(see below). Separate scalar, activation and cold-initialization saved states
check the underlying operators and all four aerosol backgrounds. Runtime tables
and fixtures are tracked; the actual table providers and CCNTY are part of the
compiled-program cache key.

Original WRF has material initialization and timestep scratch sensitivity,
and a negative cube-root intermediate that its existing zero maximum
discards. These behaviors and the unchanged source witnesses are documented
in [the upstream bug record](../proofs/s2_ntu/UPSTREAM_WRF_BUGS.md). Because of
them, original WRF results can depend on uninitialized memory; the reference
uses zero-initialized scratch memory, and no new clipping operator was added.

The GPU implementation uses one Pallas program per column. Adaptive physical
microsteps, gamma convergence, aerosol Newton iteration and sedimentation
stay inside that program. Only original DOUBLE gamma and aerosol
math/table islands use fp64; prognostic and timestep storage are REAL4.
The 44-level CUDA/sm120 Triton lowering passes on CPU for the exact production
output selection (7.51 seconds measured, 6.90 MB IR; zero fp64 arithmetic
outside original DOUBLE islands). This is a compatibility check, not a speed
claim. CPU validation of the kernel has a substantial compilation cost; GPU
launch count, register/spill usage, timing and GPU memory have not been measured.

The reference spread comes from the original WRF code built without and with
fused multiply-add (O3), and from ±1-unit rounding changes to single inputs. Any
compiled port value outside that spread is reported as a deviation; the spread
is never widened to fit the port. The 24-hour forecast comparison with the
original CPU WRF is the deciding check.

The compiled CPU Pallas public-output run has zero nonfinite values over all
8,568 observations. It has 313 values outside the fixed spread from 134
reference runs; these are recorded as deviations.
The maximum mass-field excess is 8.43e-9 kg/kg, the theta excess is
1.22e-4 K and the RAINNC excess is 1.49e-8 mm. All values and limits are in
[the reproducible deviation record](../proofs/s2_ntu/COMPILED_DEVIATIONS.json).
The original WRF compiler spread itself is large in some particle fields.

The timings of this CPU run are not usable because the machine was suspended
during it. Its finite results and static compiled buffers are valid: argument/output/temp buffers were
1,604/1,732/49,112 bytes for one 12-level column; process peak RSS was
17.5 GiB. These CPU interpreter observations cannot establish GPU speed or
memory use. The selected
scheme adds 17 REAL volume arrays and four surface arrays to the retained
State, including hail and the NTU-specific moments/totals (about 5.06 MiB
for 42x42x44, excluding transport scratch). The default scheme adds none.

A separate CPU caller test also passes: one 4x4x24
NTU call, including all adaptive microsteps, took 4.49 seconds to lower,
79.83 seconds to compile at LLVM opt0, and 11.48 seconds to execute.
Process peak memory was 6.87 GiB (zero swaps).
[The record](../proofs/s2_ntu/CPU_COMPILATION.json) lists the exact source
and settings. Timesteps, sedimentation substeps, columns and levels are not
Python-unrolled. The large shared physics graph has a substantial CPU
compiler cost; this result does not predict GPU compiler memory or speed.

A two-step CPU continuation on a real Swiss case is finite and restores the
whole model state exactly, but its ice fields go slightly negative after the
Runge-Kutta step, before any clean-up, driven by the sixth-order diffusion. Full
captures reach -1.32e-10 kg/kg in ice mass and -954.6/kg in ice number, about
1.5% of their column maxima; first samples substantially understated them.
The original WRF sixth-order diffusion reproduces this behavior on the same
full-grid inputs. It is not rounding noise, and no new clipping was added.
This is a known issue for this case; the [known-issue entry](../KNOWN_ISSUES.md#ntu56-development-case-scoped-f2-source-diffusion-undershoots)
records the measured scope (`diff_6th_opt=2`, dt=18 s, CCNTY=2, legacy code path).
The [qualification record](../proofs/s2_ntu/QUALIFICATION.json) keeps the
failed rounding test. GPU speed and memory use have not been measured, and
this finding does not cover other cases.
