# Goddard option-5 CPU qualification

The supported caller is paired `ra_lw_physics=ra_sw_physics=5`, Thompson MP8,
`use_mp_re=1`, original climatological ozone, and no aerosol/topographic/solar
interpolation extensions. Both operational loaders bind this selection. Rates
and fluxes are held at `radt` cadence; history does not re-solve radiation.

References use WRF 4.7.1 `phys/module_ra_goddard.F`, SHA256
`0917531726cbed5a128e05c45e70ee292c209dbcd1ae2bad17db6e46b8d3a5ad`.
`build_helpers.py`, `build_lw.py`, `build_sw.py`, and `build_wrapper.py` build in
a lane-private directory against `<USER_HOME>/src/wrf_pristine/WRF`. The public
wrapper oracle uses original module bytes and the public LUT initializer.
Manifest files record compiler flags, driver/source hashes, savepoint hashes
and frozen tolerances. Solver shims only expose the original private routine.

Regimes: helper interpolation spans six hydrometeors and 28 climate columns;
LW has six clear/cloud/temperature/moisture cases; SW has eight clear/cloud,
solar-angle and gas cases; the public REAL4 wrapper has clear day, cloudy day,
night, and southern high-albedo cases. Public output includes tendencies,
surface/TOA fluxes, COD, cloud-top pressure, direct/DNI/diffuse sunlight.

`literal_byte_probe.py TREE OUT.json` requires CPU
`XLA_FLAGS='--xla_cpu_max_isa=AVX --xla_disable_hlo_passes=algsimp'` and enforces
148/148 public REAL4 values BYTE. `build_wrapper.py ... --fma` builds an
independent original-source native-ISA/FMA reference. Passing its NPZ as the
third probe argument reports the ordinary compiled error and compiler spread,
while retaining the same frozen oracle gates. The measured compiled SW-rate
excess is disclosed in [the scheme notes](../../docs/goddard-radiation.md).
`mutant_probe.py TREE OUT.json` replaces only the approved zero `taux` operand
with one, and requires a real numerical assertion with zero setup errors.

The original undefined read and exact-equator out-of-bounds climate access are
documented in [UPSTREAM_WRF_BUGS.md](UPSTREAM_WRF_BUGS.md). The cloudy sensitivity
is material, so admission remains experimental. No pristine physics body is
edited; the reference uses compiler-defined initialization.

`component_bench.py TREE OUT.json` records CPU compiled memory and nine warm
samples on a 128-column tile made from the original wrapper cases. It also
checks mixed day/night batching against the original outputs. GPU execution,
E41, GPU memory/throughput, and full CPU-WRF forecast D6 remain pending.

`coupled_smoke.py TREE OUTDIR --diagnose` uses the authentic Swiss native loader
and runs three steps (42×42×44, dt18, radiation cadence33). Its raw strict-gate
receipt retains the step-3 `dynamics.qc` event. The separate manager disposition
is **matched control identical**: one out-of-range and one repair also occur in
the matched release ra4/4 control at the outer ring (F2 E213/D27). Goddard-specific
pre-zero SW counter remains zero; no Goddard gate is relaxed.

Artifacts and append-only experiment rows G00 onward are under
`<USER_HOME>/wrf_gpu2_lanes/s2-goddardrad/` and the main kernel experiment ledger.
CPU commands use `JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0`, allowed CPU affinity,
nice19 and `timeout`. Check `/tmp/wrf_gpu2_quiet` on the command line before
each heavy arm. Delete disposable build/cache/snapshot trees after use.
