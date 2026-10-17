# Experimental CPU-only FLG and GFDL radiation

Select paired `ra_lw_physics=7, ra_sw_physics=7` for Fu–Liou–Gu (FLG), or
paired `99/99` for GFDL-Eta. Set `JAX_PLATFORMS=cpu` before starting Python.
GPU selection is refused before model initialization with
`native GPU column kernel pending`. Native Pallas column kernels are a
v0.4.x follow-up; these options make no GPU performance claim.

The qualified operational caller uses Thompson `mp_physics=8`,
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

The pristine-WRF routine evidence is in `proofs/s2_flggfdl/receipts/`:

- FLG: all 121 g-points and native transfer families, 40 complete original
  columns byte-exact under literal REAL4 evaluation, then 48 complete calls
  with source phy_prep interface temperatures, allocated zero upper QV
  padding and midlatitude/polar season cases byte-exact (FG62).
- GFDL: 40 complete RADTN calls × 17 output fields byte-exact, including
  accumulators/counters; the product caller remains byte-exact (FG52/61).
  All 28 grid-dependent initialization fields are byte-exact (FG59).
  SOLARD is byte-exact on 384 date/hour cases; compiled CPU results pass
  the 16-ULP gate without fp64 (FG56). ETARA's paired held-rate mapping is
  byte-exact (FG57).

These are independent native routine comparisons on retained structured WRF
profiles with explicitly disclosed constructed caller operands and additional
regime cases, rather than replays of complete CPU-WRF forecasts. Literal
REAL4 follows WRF operation order; compiled CPU arithmetic has small documented
deviations beyond the strict component tolerance. The full GFDL CPU artifact
had maximum SW flux deviation 0.0181 W/m² and SW heating deviation
7.52e-8 K/s (FG53); FLG's largest disclosed LW TOA deviation was 0.00962 W/m²
(FG17). These excesses remain pending D6 and are not described as strict passes.
Only WRF's three declared RADTN DOUBLE scalars use fp64 in GFDL arithmetic;
the FG53 audit found no computational fp64 operations outside those islands.

CPU reference evaluation bounds scratch to one column rather than allocating
GFDL's matrix work across the whole domain. FG53 measured 185,512 temporary
bytes for one 44-level RADTN CPU artifact and a 35.76 s CPU compile. The
self-contained CO2 helper measured 0.71 s for cold build plus initialization
and 78 microseconds for a cached host provider (FG59). These are component CPU
measurements, not forecast throughput, GPU memory or executable readiness claims.
The SGS-corrected Swiss-shape CPU artifacts compiled in 34.15 s (FLG) and
42.06 s (GFDL), with temporary buffers of 5,307,528 and 3,515,632 bytes
(FG89/91). FLG scratch increases by 931,328 bytes from FG71; GFDL scratch is
unchanged. The artifacts consume 620,928/310,464 additional bytes of existing
SGS input arrays. Timing environments differ, so these are not causal speed
comparisons. FLG has zero computational fp64 operations; all nine in GFDL
map to the three native DOUBLE islands, with zero outside them.

WRF keeps `RSWTOA` and `RLWTOA` as internal fields, separate from `OLR` and the
RRTMG history flux fields. The output adapter preserves that distinction;
it does not invent an OLR value from the legacy solver's internal TOA flux.
FLG's original whole-tile `USWTOP` assignment is retained and disclosed.
All new baked tables, CO2 providers and initial calendar constants participate
in JAX/AOT keys, with collision regressions for changed values.

CPU coupled smoke is integration evidence. Full coupled-forecast fidelity/D6
qualification remains pending. The release-default RRTMG program is unchanged;
the post-glacier Swiss reference identity is `f672a5739cb9…`, PROD d01 `6fc715a2`.

The original three-step Swiss FLG smoke (radt=10 minutes) retained finite state and byte-identical held
radiation. Step 3 reported one QC out-of-range event and its repair at the outer
boundary `(k=15, y=41, x=8)`, value `−1.159596e−36`; its post-RK minimum elsewhere
was `−5.998731e−18`. The matched RRTMG control had zero guard events. The manager
classified this as F2 round-off from the different heating trajectory, with the
same boundary cell also observed in the Goddard lane. This is a disclosed
trajectory difference, rather than an exact matched-control event. No guard or
tolerance changed. The formal disposition and all raw counts are retained in
`FG66_MANAGER_ROUNDOFF_DISPOSITION.json` and `FG66_flg_coupled.json`.

The operational caller consumes the shared WRF radiation views of QC and QI,
including MYNN SGS condensate at the native thresholds (radiation_driver
1401–1430). Other PBL selections use the existing policy that disables this
MYNN merge. Eight source-bound seam controls cover both schemes, both PBL
policies and initial/later calls; deleting either connection produces numerical
failures. The merged arrays are reused rather than recomputed.

The transport test suite retains 17 known strict compiled diagnostics as
explicit, band-specific expected failures (IR 7 and 9–18; SW 1–6), pending D6.
Their original native assertions and bounds remain intact. Separate mandatory
checks verify all 18 bands are finite and unchanged against a frozen CPU
compiled regression baseline, and reject zero/NaN corruption. That baseline
is JAX regression evidence only, never WRF fidelity or a relaxed tolerance.

After the SGS correction, both schemes completed a native REAL Swiss cadence
stress (dt=18 s, radt=0.6 minutes, STEPRA=2): three finite steps, zero guard
events, two radiation solves, held diagnostics byte-identical between solves,
and zero output radiation calls (FG88/90). Nonzero SGS liquid and ice were
present before the later refresh. The harness asserts fp32 carry and records
precision flags before stepping. Earlier FAST_DEFAULTS=0 repair arms remain
explicitly labeled legacy CPU integration evidence, separate from these native
gates. The final repair receipt is `receipts/CRITIC_REPAIR_HANDOFF.json`.

Independent native-envelope disposition (critic `rv-s2-flggfdl`, reviewed
`dc5139f7e`): 17 strict bands contain 3,254 values outside the original strict
allowance. Seven strict-deviation bands (SW5/6; IR11/12/13/14/18) fit the
independently measured native FMA envelope plus the original allowance. Ten
other bands retain 285 values outside even that envelope, with maximum band
error 0.0013961792 W/m² and maximum source-bound ratio 12.8554. These ten
bands use only the small documented experimental deviation/D6-pending policy;
they are not source-envelope passes. The original bound is the matched
profile/field native FMA-vs-no-FMA Linf spread + 16 reference ULP + 3e-6 W/m².

| Residual band | Strict excess values | Native-envelope excess values | Max error (W/m²) | Max bound ratio |
| --- | ---: | ---: | ---: | ---: |
| SW1 | 1216 | 50 | 0.000285744667 | 1.55095 |
| SW2 | 334 | 45 | 1.30188782e-05 | 2.2085 |
| SW3 | 179 | 7 | 1.60932541e-05 | 1.28405 |
| SW4 | 44 | 20 | 1.55568123e-05 | 2.95477 |
| IR7 | 125 | 47 | 6.05583191e-05 | 1.93913 |
| IR9 | 108 | 46 | 0.00106716156 | 1.79861 |
| IR10 | 104 | 5 | 0.000321388245 | 1.49681 |
| IR15 | 102 | 21 | 0.000396728516 | 1.0972 |
| IR16 | 201 | 13 | 0.0013961792 | 12.8554 |
| IR17 | 121 | 31 | 9.53674316e-05 | 4.1926 |

The unchanged full-field FLG evidence also retains LW TOA error
0.0096130371 W/m² versus native FMA spread 0.0011901855 W/m², and maximum
heating error 4.4481567e-7 K/s. GFDL has its own distinct residuals,
0.0181 W/m² SW flux and 7.52e-8 K/s SW heating. These remain D6-pending
deviations. Exact per-band counts, independent bound, dispositions and all three
fixture hashes are in the [tracked disposition](../proofs/s2_flggfdl/receipts/FG93_flg_compiler_disposition.json)
and the [unaltered critic bound receipt](../proofs/s2_flggfdl/receipts/FG93_flg_native_bound.json).

The critic's remaining compiled-zero hard FAIL is an **intentional mutation
kill**: its harness replaces actual IR outputs with zeros. The hard regression
case rejects those outputs with a numerical AssertionError (1 FAIL / 1 strict
XFAIL, zero setup errors). The unmodified candidate independently had
89 PASS / 17 XFAIL / 0 FAIL / 0 ERROR. This is no unresolved clean-candidate
failure; the hard regression must keep failing on corrupted outputs. The
[mutant harness](../proofs/s2_flggfdl/receipts/FG93_compiled_zero_mutant.py),
mutant result,
clean candidate result,
and [critic verdict](../proofs/s2_flggfdl/receipts/FG93_VERDICT_REPAIR_dc5139f7e.md)
are retained. No strict assertion, native bound, XFAIL band, solver, or guard
changed for these metadata corrections, and no numerical validation was repeated.
