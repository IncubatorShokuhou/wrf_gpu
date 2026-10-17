# P3 microphysics (mp_physics=50/51/52/53)

**Status in v0.4.0: Validated 24 h.** Passed a 24-hour comparison with the original CPU WRF in the v0.4.0 release check, in one tested single-domain configuration. Other combinations and nested domains are not validated.

The port implements the original WRF P3 wrappers, including
column processes, adaptive sedimentation, freezing, four Registry variants,
particle-property transport and root flow-dependent boundaries. Select the
variant in `namelist.input`. Live nests remain refused before compute.

| Option | Prognostic particle additions |
|---|---|
| 50 | Rime mass and volume; prescribed cloud number |
| 51 | Rime mass and volume; predicted cloud number |
| 52 | Two ice categories, each with mass, number, rime mass and volume |
| 53 | Rime mass and volume; advected sqrt(N*Z) third-moment representation |

All selected particle/history leaves use REAL32 and are allocated only for
that variant. Old theta/vapor are held physics history, not advected scalars.
Cloud/ice effective radii are held in State. For RRTMG with `use_mp_re=1`,
physics initialization seeds the original 2.49/4.99um backgrounds; other
selections retain Registry zero until the first microphysics call.
WRF's P3 caller moves total ice to the snow optics channel, using the predicted
ice radius there; it clears the ordinary ice optics channel. The two-category
case first sums both ice masses. P3 does not carry a separate snow radius.

P3 particle variables appear in history with their original Registry names and
units. Both operational pickle and NetCDF restart retain every selected
particle, old-physics field and radius; missing active carries fail closed.
The QZI backend boundary rounds the square/product to REAL before division or
sqrt, preserving subnormal intermediates. CPU emulation uses a wide temporary
container for that rounding; native CUDA uses non-FTZ REAL instructions and
adds no DOUBLE island or State allocation.

Column checks against the original WRF code use single-precision (REAL4)
columns with clear, warm, cold, rimed, mixed-phase, two-category and
third-moment cases. The literal no-FMA execution of the port matches every
saved field bit for bit for all four variants and three calls. The compiled path passes the independently measured
original-WRF FMA spread and a source-derived cloud-radius discontinuity bound;
original strict tolerance scores remain in the evidence. See
[compiled bound](../proofs/s2_p3/COMPILED_BOUND.md) and
[compiler initialization audit](../proofs/s2_p3/UPSTREAM_WRF_BUGS.md).
Lookup/property fixtures and original wrapper manifests bind pristine source,
Registry, compiler flags, table hashes, units and active-regime census under
`proofs/s2_p3/`. Radiative caller tests use unchanged pristine LW/SW blocks.

The native entry puts one complete column, including its adaptive loops,
inside one Pallas program. Tables are host-cached immutable NumPy arrays and
passed as device operands; there is no per-step host lookup or transfer.
Selected table bytes are approximately 0.33MB for 50/51, 2.33MB for 52 and
3.34MB for 53, plus small factorial/DERF constants. New selected particle and
old-physics carries use 16/16/32/20 bytes per cell respectively, with two radii
adding 8 bytes per cell. The default mp=8 adds no P3 arrays or kernels; the
default forecast program is unchanged. GPU launch count, peak GPU memory and
speed have not been measured.

```sh
JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 python proofs/s2_p3/literal_cpu.py --out literal.json
JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 python proofs/s2_p3/run_column_gate.py \
  --wrf-spread /path/to/independent/P348_WRF_contract --out compiled.json
```
