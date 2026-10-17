# FLG and GFDL radiation (CPU only)

**Status in v0.4.0: CPU only (experimental).** `gpuwrf run` refuses
`ra_lw_physics`/`ra_sw_physics` = 7 and 99 on the GPU; GPU kernels are not
available yet.

Select paired `ra_lw_physics=7, ra_sw_physics=7` for Fu–Liou–Gu (FLG), or
paired `99/99` for GFDL-Eta. Set `JAX_PLATFORMS=cpu` before starting Python.
A GPU selection is refused before model initialization with the message
`native GPU column kernel pending`. These options make no GPU performance claim.

The supported operational caller uses Thompson `mp_physics=8`,
`cu_physics=0`, Noah-MP `sf_surface_physics=4`, and supplied WRF grid/surface
statics. Mixed radiation pairs, other microphysics/land pairings, convection,
urban/lake coupling, aerosol feedback, topographic radiation and solar
interpolation are refused. GFDL radiation does not require GFDL surface-layer
physics: WRF's ETARA call has no such mandatory pairing.

Both pipelines bind the selected radiation pair and check that it was honoured.
Each radiation cadence performs one paired solve. The theta rates, surface
forcing and native diagnostics are held between calls and at output. GFDL also
retains its native cloud counters, fractions and interval convection fields.
Restart readers reject missing fields, changed selections and incompatible
shapes/dtypes instead of cold-starting selected physics.

GFDL needs GNU Fortran (`gfortran`, or `FC=/path/to/gfortran`) on its first use.
The package contains WRF's unchanged CO2 initialization routines, constants and
three input tables; an installed WRF tree is unnecessary. A bounded serial
helper runs once before stepping, then only read-only REAL4 tables enter the
solver. Its executable cache defaults to `$XDG_CACHE_HOME/gpuwrf/gfdl_init`
(or `~/.cache/gpuwrf/gfdl_init`); `GPUWRF_GFDL_INIT_CACHE` overrides it.
No host callback or compiler process runs inside a timestep.

The evidence against the original WRF routines is in `proofs/s2_flggfdl/receipts/`:

- FLG: all 121 g-points and native transfer families, 40 complete original
  columns byte-exact under literal REAL4 evaluation, then 48 complete calls
  with source phy_prep interface temperatures, allocated zero upper QV
  padding and midlatitude/polar season cases byte-exact.
- GFDL: 40 complete RADTN calls × 17 output fields byte-exact, including
  accumulators/counters; the product caller remains byte-exact.
  All 28 grid-dependent initialization fields are byte-exact.
  SOLARD is byte-exact on 384 date/hour cases; compiled CPU results pass
  the 16-ULP limit without fp64. ETARA's paired held-rate mapping is
  byte-exact.

These are independent native routine comparisons on retained structured WRF
profiles with explicitly documented constructed caller operands and additional
regime cases, rather than replays of complete CPU-WRF forecasts. Literal
REAL4 follows WRF operation order; compiled CPU arithmetic has small documented
deviations beyond the strict component tolerance (see the implementation notes
below). Only WRF's three declared RADTN DOUBLE scalars use fp64 in GFDL
arithmetic; an audit found no computational fp64 operations outside those
islands.

CPU reference evaluation bounds scratch to one column rather than allocating
GFDL's matrix work across the whole domain. One 44-level RADTN CPU artifact
used 185,512 temporary bytes and compiled in 35.76 s. The self-contained CO2
helper measured 0.71 s for cold build plus initialization and 78 microseconds
for a cached host provider. These are component CPU measurements, not forecast
throughput, GPU memory or start-up time claims.
The SGS-corrected Swiss-shape CPU artifacts compiled in 34.15 s (FLG) and
42.06 s (GFDL), with temporary buffers of 5,307,528 and 3,515,632 bytes.
FLG scratch is 931,328 bytes larger than in the earlier version; GFDL scratch
is unchanged. The artifacts consume 620,928/310,464 additional bytes of existing
SGS input arrays. Timing environments differ, so these are not causal speed
comparisons. FLG has zero computational fp64 operations; all nine in GFDL
map to the three native DOUBLE islands, with zero outside them.

WRF keeps `RSWTOA` and `RLWTOA` as internal fields, separate from `OLR` and the
RRTMG history flux fields. The output adapter preserves that distinction;
it does not invent an OLR value from the legacy solver's internal TOA flux.
FLG's original whole-tile `USWTOP` assignment is retained.
All new baked tables, CO2 providers and initial calendar constants are part of
the compiled-program cache keys, with regression tests for changed values.

Short CPU coupled runs show that the schemes are wired into the model correctly;
no full forecast comparison with the original CPU WRF has been done. The default
RRTMG program is unchanged.

A three-step FLG run on a Swiss case (radt=10 minutes) kept a finite state and
byte-identical held radiation. Step 3 reported one QC out-of-range event and its
repair at the outer boundary `(k=15, y=41, x=8)`, value `−1.159596e−36`; its
post-RK minimum elsewhere was `−5.998731e−18`. The matched RRTMG control had zero
guard events. This is treated as rounding noise from the different heating
path; the same boundary cell also appears with Goddard radiation. No guard or
tolerance changed. All raw counts are kept in `proofs/s2_flggfdl/receipts/`.

The operational caller consumes the shared WRF radiation views of QC and QI,
including MYNN SGS condensate at the native thresholds (radiation_driver
1401–1430). Other PBL selections use the existing policy that disables this
MYNN merge. Eight source-bound seam controls cover both schemes, both PBL
policies and initial/later calls; deleting either connection produces numerical
failures. The merged arrays are reused rather than recomputed.

After the SGS correction, both schemes completed a native REAL Swiss cadence
stress test (dt=18 s, radt=0.6 minutes, STEPRA=2): three finite steps, zero guard
events, two radiation solves, held diagnostics byte-identical between solves,
and zero output radiation calls. Nonzero SGS liquid and ice were present before
the later refresh. The harness asserts fp32 carry and records precision flags
before stepping.

## Implementation notes

Compiled CPU results for FLG differ from the strict reference in 17 spectral
bands (IR 7 and 9–18; SW 1–6), with 3,254 values outside the original strict
allowance. Seven of these bands (SW5/6; IR11/12/13/14/18) stay within the spread
that the original WRF code itself shows with and without fused multiply-add,
plus the original allowance. The other ten bands keep 285 values outside even
that spread; the largest band error is 0.0013961792 W/m². FLG's full-field LW TOA
error is 0.0096130371 W/m² (versus a native FMA spread of 0.0011901855 W/m²) and
its maximum heating error is 4.4481567e-7 K/s. GFDL's largest deviations are
0.0181 W/m² in SW flux and 7.52e-8 K/s in SW heating. These are documented
deviations, not passes. The test suite marks the 17 bands as expected failures,
keeps their original assertions and limits, and separately checks that all 18
bands are finite and unchanged against a fixed CPU regression baseline. That
baseline is a regression check only, never WRF fidelity or a relaxed tolerance.
Per-band counts and limits are in the
[tracked disposition](../proofs/s2_flggfdl/receipts/FG93_flg_compiler_disposition.json).
