# Experimental SSiB (sf_surface_physics=8)

This opt-in JAX port transcribes all 31 routines and 27 embedded DATA tables of
pristine WRF 4.7.1 `phys/module_sf_ssib.F`. Runtime execution needs no Fortran,
external SSiB table, host callback, or transfer inside the timestep.

## Admitted configuration

Both native/nested and daily/replay loaders bind prepared SSiB inputs with
`sf_surface_physics=8`, `sf_sfclay_physics=1`, `bl_pbl_physics=1`,
`mp_physics=8`, `ra_sw_physics=1`, `ra_lw_physics=4`, `num_soil_layers=3`,
`fractional_seaice=1`, no lake/urban model and no positive SEAICE. Soil geometry
is WRF `init_soil_depth_8`: ZS=[0,.05,1.05] m, DZS=[.05,1,0] m. The loader
refuses Noah four-layer inputs and never interpolates them. USGS/SSIB/MODIS
vegetation maps use the pristine embedded conversion/parameter tables. Fresh
startup also runs the WRF LANDUSE initialization from the tracked pristine
`io/ssib_landuse.py` values: ALBEDO/EMISS, MAVAIL and initial ZNT. These are
physics-init outputs, not required real.exe input variables. `usemonalb` and
custom LANDUSE.TBL files are refused. Native unmodified LANDUSE_INIT: 356
rows/2,136 output fields BYTE across four maps, seasons and hemispheres.

SW=3/4 is refused: the retained radiation interface lacks SSiB's four absolute
visible/NIR direct/diffuse components. No estimated spectral split is used.
Sea-ice arithmetic is present in the transcription but is not admitted or
qualified. Land ice follows WRF's vegetation tables. Other pairings remain
refused until their caller gates are supplied.

The caller retains 101 leaves (100 REAL4, ISNOW INTEGER4), including all 48 snow
scalars, canopy/soil state and Registry diagnostics, plus three held REAL4
radiation planes. Revised-MM5 runs before SSiB; YSU consumes actual SSiB
HFX/QFX/UST/BR and the paired surface-layer FM/FH. WRF's actual T2=TSK writeback
is retained. Precipitation forcing uses the WRF-order held precipitation carry.
Dudhia SW and RRTMG LW each run once per refresh. COSZEN follows WRF
XTIME+RADT/2 in REAL minutes, while radconst retains the current time. Registry r-only snow fields
are restart leaves; the 15 SIB history fields use Registry names/units/staggers.
All three public restart formats validate selection, schema, shapes, dtypes,
finiteness and held radiation before restore; active missing groups fail.

## Numerical evidence and disclosure

`native_reference.npz` and its manifest contain original-WRF REAL4 outputs,
with unmodified source checksum and frozen tolerances. Coverage: daylight,
night, wet soil, multilayer snow, melting, snowfall and rain on snow; 400
complete column calls. `literal_gate.py` executes the same transcription with
literal REAL4/libm/no-FMA semantics: 400/400 BYTE, zero differences. The NumPy
backend is validation-only, never a product fallback or fidelity self-compare.

Compiled ordinary/cold-snow calls pass 8,775 field checks against frozen
original-WRF FMA/no-FMA spreads and independently propagated RASIT5 input-ULP
bounds. The other 300 calls have **56 field comparisons beyond those bounds**
(out of 30,000), maximum flux error 0.05615234 W/m2 and temperature error
0.00067139 K. Five of the 56 are QFX residuals caught by tightening its
unit-specific mass-rate bound (2mW/m2 / WRF HLAT=2.52e6 J/kg); they match
already disclosed latent-heat cells. Maximum QFX error is 1.67165e-8 kg/m2/s. See `compiled_residuals.json` and
[UPSTREAM_WRF_BUGS.md](UPSTREAM_WRF_BUGS.md). These are disclosed experimental
compiled deviations under the manager's standing policy; the original
precision tolerances were not widened to observed errors. GPU and full
coupled-forecast qualification remain pending; D6 against CPU-WRF decides.

Pristine UPDAT1 reads uninitialized local SNOWW(2) on its ISNOW=0 path. The
oracle uses compiler-defined zero initialization without changing source bodies;
the port mirrors that policy. Cold-matrix zero/one sensitivity is 7.8444e-12 m
runoff (all other 9,674 values identical). This is limited coverage, not a claim
about every forecast. WRF fatal branches are observable nonfinite failures,
not silently ignored or clamped.

## Reproduction and cost

CPU commands require the repository QUIET guard, lane core pinning, nice 19,
`JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0`, an immutable tree and a timeout.
Run `python proofs/s2_ssib/literal_gate.py TREE OUTPUT.json`; run the targeted
SSiB tests with lane-local pytest basetemp. `audit_oracle.py` stages its own
pristine build and verifies source SHA; zero mode supplies deterministic GNU
initialization. Compiler binaries/build dirs are disposable lane artifacts.

All evolving SSiB storage is resident: 404 bytes/cell land carry plus 12
bytes/cell radiation, and the existing held WSPD State plane. This is incremental
SSiB storage, not total forecast memory. The six required history planes and
five precipitation planes add 44 logical bytes/cell where those shared driver
containers are not already allocated (464 bytes/cell including this context). No default-path leaves or solves are
added. Final complete scalar CPU API (SS59): compile 53.506 s, arguments 500 B,
output 1,212 B, temporary 55,352 B. Bounded snow loops reduce traced operations 181,756→99,253, scalar temporary
memory 97,784→55,352 B and measured compile 80.244→53.506 s. All 400 compiled
outputs are BYTE-identical to the pre-optimization artifact (regression only);
final literal REAL4 remains BYTE vs original WRF. All 400 compiled calls are finite; ordinary
and cold-snow tests pass, and the three wet/melt groups are explicit expected
numerical failures matching the frozen residual artifact exactly (regression
identity only, not a fidelity self-compare). No error-derived precision bound
was substituted.

The final real-loader CPU plumbing smoke (SS68, Swiss 42x42x44, dt=18 s) completed two
steps: load 13.778 s, first step including compilation 295.572 s, warm second
step 5.974 s; all carry leaves finite, pre-mask guards=0. These are CPU smoke
costs, not GPU throughput or a CPU-WRF forecast comparison. The full executable
uses a single stable static-bundle identity, avoiding duplicate compilation.
Measured whole-process peak CPU RSS was 9,096,708 KiB (8.68 GiB), including
compilation. A GPU-memory comparison is not claimed. GPU
launch count, E41/cubin, VRAM and throughput remain unmeasured in this no-GPU
wave; generated scalar loops/predication may need GPU tuning before release.

`cpu_smoke.py` uses private Swiss Noah input converted to three test soil levels
solely to test real loader/production-step plumbing; normal WRF LANDUSE startup
supplies the radiative surface properties.
It is not a real.exe-prepared SSiB forecast or CPU-WRF forecast fidelity claim.
