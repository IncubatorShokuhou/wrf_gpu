NTU is an opt-in experimental microphysics option: set `mp_physics=56` in
`&physics`. `ccnty=1/2/3/4` selects the original WRF aerosol background; its
Registry default is 2. Qualification uses original WRF 4.7.1
`module_mp_ntu.F` and the `module_microphysics_driver.F:1586` argument map.
GPU and coupled forecast qualification are pending.

The port carries all seven moist fields and 21 scalar fields. Six scalars are
hydrometeor number concentrations. The other fifteen are dry/total/cloud/rain
condensation nuclei, potential ice nuclei, and ten ice shape/volume/moment
fields. These fields are transported with the appropriate moist/scalar
advection options, use available root boundary records, appear under the
original Registry history names, and survive lossless restart. Live nests
remain refused pending a coupled qualification.

Cold initialization calls the original aerosol equations using the dycore's
dry-air specific volume, staggered geopotential and actual vapor. Restarts
retain the saved aerosol and moment state. History RAINNC and SNOWNC retain
the original REAL accumulation order. The land coupling retains original
public RAINNCV/SNOWNCV/GRAUPELNCV/HAILNCV rather than differences of rounded
totals. NTU's public RAINNCV is rain-only and SNOWNCV is snow-only, despite
the broader accumulated totals; this upstream partition is preserved and
disclosed. No effective radii or radar fields are fabricated.

Literal REAL4 and REAL8 qualification covers 12 warm, cold, mixed-phase,
evaporating and ice-supersaturated columns, with all prognostics active across
the corpus and two evolving calls. All 8,568 public values per precision are
byte-identical to pristine WRF under the project's compiler-zero policy.
Separate scalar, activation and cold-initialization savepoints qualify the
underlying operators and all four aerosol backgrounds. Runtime tables and
fixtures are tracked; the actual bound providers and CCNTY enter AOT keys.

Original WRF has material initialization and timestep scratch sensitivity,
and a negative cube-root intermediate that its existing zero maximum
discards. These behaviors and the unchanged source witnesses are documented
in [the upstream bug record](../proofs/s2_ntu/UPSTREAM_WRF_BUGS.md). They make
this an experimental option; zero-initialized upstream scratch is the stated
reference policy, and no new clipping operator was added.

The GPU implementation uses one Pallas program per column. Adaptive physical
microsteps, gamma convergence, aerosol Newton iteration and sedimentation
stay inside that program. Only original DOUBLE gamma and aerosol
math/table islands use fp64; prognostic and timestep storage are REAL4.
The 44-level CUDA/sm120 Triton lowering passes on CPU for the exact production
output selection (7.51 seconds measured, 6.90 MB IR; zero fp64 arithmetic
outside original DOUBLE islands). This is a compatibility check, not GPU execution or a speed
claim. CPU Pallas validation has a substantial compilation cost; GPU
launch count, register/spill usage, E41, timing and VRAM remain unmeasured.

Independent original-WRF compiler controls are fixed at no-FMA/O3 native-FMA
and single-input +/-1 ULP perturbations. Any compiled port value outside that
enclosure must be reported as an experimental deviation; the enclosure is
never expanded to fit the port. D6 forecast comparison remains authoritative.

The compiled CPU Pallas public-output run has zero nonfinite values over all
8,568 observations. It has 313 values outside that fixed 134-control
enclosure; this is an experimental deviation record, not an enclosure pass.
The maximum mass-field excess is 8.43e-9 kg/kg, the theta excess is
1.22e-4 K and the RAINNC excess is 1.49e-8 mm. All values and bounds are in
[the reproducible deviation receipt](../proofs/s2_ntu/COMPILED_DEVIATIONS.json).
The original WRF compiler spread itself is material in some particle fields.
The candidate stays unqualified for coupled forecast accuracy until D6.

The CPU Pallas numerical arm spanned host hibernation, so its compile,
execution and elapsed timings are invalid (E60). Its finite observations
and static compiled buffers remain valid: argument/output/temp buffers were
1,604/1,732/49,112 bytes for one 12-level column; process peak RSS was
17.5 GiB. These CPU interpreter observations cannot establish GPU speed or
memory use. The selected
scheme adds 17 REAL volume arrays and four surface arrays to the retained
State, including hail and the NTU-specific moments/totals (about 5.06 MiB
for 42x42x44, excluding transport scratch). The default scheme adds none.

The required CPU JAX caller smoke also passes independently: one 4x4x24
NTU call, including all adaptive microsteps, took 4.49 seconds to lower,
79.83 seconds to compile at LLVM opt0, and 11.48 seconds to execute.
Process peak RSS was 6.87 GiB (zero swaps), exceeding the normal 6 GiB
worker cap; the manager authorized a 12 GiB exception for this singleton.
[The receipt](../proofs/s2_ntu/CPU_COMPILATION.json) records the exact source
and settings. Timesteps, sedimentation substeps, columns and levels are not
Python-unrolled. The large shared physics graph has a substantial CPU
compiler cost; this result does not predict GPU compiler memory or speed.

The real Swiss two-step CPU continuation is finite and round-trips the whole
carry, but its pre-mask RK ice fields have source-driven undershoots. Full
captures reach -1.32e-10 kg/kg in ice mass and -954.6/kg in ice number, about
1.5% of their column maxima; first samples substantially understated them.
Pristine WRF's frozen sixth-order diffusion reproduces this class on the
same full-grid inputs. It is not labelled round-off, and no new clipping was
added. The manager accepted this case-scoped source-diffusion F2 known issue
on 2026-10-14; a full original RK-chain comparison is not required for that
disposition. The [known-issue disclosure](../KNOWN_ISSUES.md#ntu56-development-case-scoped-f2-source-diffusion-undershoots)
records the measured scope (`diff_6th_opt=2`, dt=18 s, CCNTY=2, FAST0).
The [qualification ledger](../proofs/s2_ntu/QUALIFICATION.json) retains the
failed round-off test. GPU/E41, performance/VRAM and full D6 remain pending;
this acceptance does not qualify other cases or establish global F2 clearance.
