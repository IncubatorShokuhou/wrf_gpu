# Number-scalar transport — v0.4 development

WDM5/6/7 (`mp_physics=14/16/26`) now transport cloud and CCN numbers (`Nc/Nn`),
and Morrison (`10`) transports snow and graupel numbers (`Ns/Ng`). WDM's rain
number `Nr` is transported exactly once; its inactive ice-number field `Ni`
no longer enters transport or MYNN scalar forcing. On the starting source,
`State.Ni` was materialized even for WDM, so `Nr` could already enter the old
Ni/Nr pair. The omissions were Nc/Nn and Morrison Ns/Ng.

The input reader loads `QNSNOW/QNGRAUPEL/QNCLOUD/QNCCN`, and WDM7 preserves
supplied `QHAIL` during native initialization. Number fields remain
in the existing State buffers and restart payloads. Active restarts missing
their required number fields fail instead of silently cold-starting them.
The shared namelist binder honours `ccn_conc` in both pipelines (WRF default
`1e8`). WDM QNN inflow uses pristine WRF's prescribed CCN value, with interior
copy at outflow; first-step WDM CCN initialization occurs after RK transport
and resumed steps retain their CCN state.

These schemes are **single-domain, CPU-operator-qualified**. Nested runs remain
refused until parent-to-child forcing is qualified. GPU and coupled-forecast
qualification remain pending. WDM7 uses the documented fp64 column island on
the REAL32 carry, as WDM5/6 already do. Morrison's raw fp32 trace defect is
fixed; its existing qualified fp64 island remains until separate numerical
qualification of the REAL column kernel. This change does not claim new
native fp32 microphysics qualification.

## Evidence

- Nine QNN boundary cases match unchanged pristine REAL4 WRF byte for byte:
  specified widths 1/2/3, CCN values 0/1e8/2.7e8, all four sides, mixed
  inflow/outflow and exact-zero normal velocity.
- Four pristine scalar-advection/RK/boundary chains cover Morrison and WDM6
  with scalar options 0/1, all three RK stages, and active constructed number
  profiles on real Swiss wind/mass/map/eta operands. The declared bound is
  32 ulps at each reference field maximum; both sides must be finite. It was
  fixed before NS02 results. This is an operator check on a cropped complete
  test domain, not a WDM/Morrison forecast compared with original CPU-WRF.
- The numerical focused subset passes 49 tests, including actual Swiss number
  input reads, restart retention/admission, both-pipeline CCN binding,
  first-own-step/resume behavior, and raw fp32 Morrison tracing. All eight
  required shared-consumer files pass. Existing original-WRF Morrison column
  oracles also pass. The one GPU-only hail allocation test is skipped on CPU.
- Deleting the forced native PD route produces a genuine numerical assertion
  failure; deleting number transport produces four genuine route assertion
  failures. Neither mutant has a setup error. The composed CLI and boundary
  consumer selection passes 55 tests.
- Final input/transport/binding/catalog checks pass 153 tests; the composed
  NSSL caller and namelist checks pass 60. All six final CLI trace hashes,
  bindings and dispatch records match the qualified candidate exactly.
- Actual Swiss CLI traces pass for 10/14/16/26, with three scalar updates and
  four boundary passes per own step and REAL32 number outputs. Mixed WDM6
  moist/scalar options 0/1 with CCN 2.7e8 use the correct separate options.
  The release Swiss jaxpr remains
  `2bf12cb8374d3a84a440e47dea4b77b0c7436108fc2ee2616bba6ec79b17962a`.
- The 75,026-byte input fixture is tracked and loads with matching
  checksum/shapes/kinds from a clean `git archive`. Its manifest records
  staggering, units, construction, source checksum and tolerance provenance.

The repaired families use the existing pristine-qualified native PD kernel
on specified domains even when its global fast flag is off. The legacy PD
fallback failed the sharp QNN inflow case; no tolerance was relaxed to admit
that result. Equal advection options retain one stacked transport call;
different options retain WRF's distinct moist/scalar selections. Horizontal
and sixth-order scalar mixing also include the repaired fields.

## Coupled smoke and limits

CPU smoke uses the real Swiss single-domain inputs, release REAL32 carry,
Pallas interpretation and a CPU PTX-rounding shim, two 18-second own steps,
resident phase census and disabled policy repairs. These are CPU execution
checks, not executions of a GPU binary or forecast-fidelity gates. The dry
Swiss smoke has zero terminal Nc/Nr in WDM and zero Morrison number fields;
active transport is covered by the independent operator fixtures above.

All four schemes have zero negative/nonfinite active-number events in the
observed REAL32 phase rows (two observations per row), zero nonfinite carry
values, and zero applied policy repairs. Morrison has zero events across all
observed fields. WDM5/6/7 shared ice-mass events remain open: 123 post-RK
negative visits, minimum
`-3.7991054e-15 kg/kg`, and five surviving outer-ring visits, minimum
`-1.318875e-22 kg/kg`. Counts, extrema, locations and coverage match the old
source exactly. That comparison is **regression/attribution only**, not WRF
truth or an all-field F2 waiver. The old inactive WDM Ni negative events
disappear. The full per-scheme smoke receipts retain the remaining events.

## Performance and memory

No GPU launch, VRAM, compile or speed measurements were made in this CPU-only
wave. No State leaf was added, and no timestep host/device transfer was added.
The ordinary root transport/boundary call counts are unchanged when the two
options match. The scalar batch grows from 8 to 10 lanes for Morrison and from
8 to 9 for WDM5/6 (9 to 10 for WDM7): WDM adds two active fields and removes
inactive Ni. Their tendency/PD/mixing temporaries and traffic increase with
that necessary WRF work. GPU peak memory and compiled fusion costs remain
unmeasured. The default release program is unchanged by the recorded identity
gate; the opt-in fp64 islands remain a qualification/performance limitation.

Reproduction tools and fixture manifest: [`proofs/s1_numbers`](../../proofs/s1_numbers/),
[`tests/v040/numbers`](../../tests/v040/numbers/). Small receipts and detailed
logs are under `<USER_HOME>/wrf_gpu2_lanes/s1-numbers/`; the final status packet
lists their hashes. No full forecast or overall scientific release clearance
is inferred from these CPU checks.
