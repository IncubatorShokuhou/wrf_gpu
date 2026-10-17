CAM-UW T2 triage
===============

The retained Swiss BL9/MP8/SF5/Noah-MP/RRTMG forecast first fails T2 at
hour 4: RMSE 1.700204 K against the original CPU-WRF, versus the frozen 1.5 K cap.
This is nighttime (SWDOWN zero). At that hour CPU-WRF has 6,762 nonzero
CLDFRA cells, whereas the GPU has zero; GLW RMSE is 50.266730 W/m².

Original WRF `share/module_check_a_mundo.F:1802–1820` disables `icloud_bl`
without MYNN. The previous GPU radiation input assembly always used MYNN's
subgrid cloud field, which CAM-UW leaves at zero, erasing resolved clouds after
the first step. The fix selects WRF `cal_cldfra1` for non-MYNN RRTMG callers.
It covers initialization, held radiation, split SW/LW and fallback output calls.
The default MYNN call supplies no additional arguments.

CAM-UW also previously supplied post-RK increments rather than WRF's raw
RUBLTEN/RVBLTEN/RTHBLTEN/RQVBLTEN/RQCBLTEN/RQIBLTEN/RQNIBLTEN at RK cadence.
The adapter now exposes these existing rate arrays to the existing RK consumers.
The r8 CAM column equations, carry schema and legacy adapter API are unchanged.
The adapter now also passes hydrostatic pressure and interfaces, as the WRF
PBL caller does, while retaining nonhydrostatic Exner/temperature independently.
It reuses the existing native hydrostatic pressure helper.

`acquire_cloud_oracle.py` extracts and compiles the verbatim pristine WRF
`cal_cldfra1` body in a private build. `cloud_oracle.json` contains 40 regimes
and results from 520 original calls, including each input's ±1-ULP response.
The numerical test uses that independently measured input-ULP bound.
The CAM column itself retains the existing 78-record pristine oracle gate.

`retained_localize.py` reads retained original CPU/GPU fields without mutation.
`cpu_forecast_probe.py` uses the production loader and timestep loop on the real
Swiss inputs, recording the first three steps and optionally h1–4 T2/TSK/GLW.
`release_identity.py` uses the common native release trace probe, with temporary
files redirected to this lane. `cloud_selection_mutant.py` forces the original
wrong cloud selection and must fail the assembly test numerically.

Corrected GPU 24-hour D6, E41 and timing/memory evidence remain required before
claiming that the forecast qualification failure is closed. The pre-existing
CAM coupling caveat (total radiative heating supplied instead of LW-only) is
outside these nighttime fixes and remains disclosed in the catalog.

Final original-reference CPU evidence (TC15, measured source pinned by local tag
`evidence/t-camuw/TC15-bb14622a1`): h1–4 T2 RMSE is
0.085831/0.087999/0.093639/0.083914 K, below the 1.5 K target. It completed 800
steps; the observed carries at steps 1/2/3/200/400/600/800 are finite.
The first three steps have zero guard candidates; the final aggregate
is 81,424 candidates and remains OPEN. This first probe retained only aggregate
counts; the current probe also retains per-field F2 locations and minima for
future runs. No numerical rerun was made just to repeat completed validation.
Strict guards are disabled, so the census's `repaired` label counts candidates,
not actual masking. The old 24-hour GPU count was 7,948; same-event equivalence
has not been established. Final GPU validation must classify its actual events.

The final rebased CPU CAM program and all 61 captured constants exactly match
the measured source (`9270203c…`), carrying forward TC15. The final native release
trace remains `2bf12cb8…`. Shared checks: 280 passed; final affected CAM checks:
15 passed. Original resolved-cloud oracle, numeric selection mutant, clean
archive, manifest validation and static transfer checks pass. Receipts live
under `<USER_HOME>/wrf_gpu2_lanes/t-camuw/` and the compact tracked receipt packet.

Manager-requested CAM-ZM seam (`c313f125e`): `return_cumulus_inputs=True`
appends `(TPERT, KPBL)`, with REAL32 kelvin and an INTEGER32 one-based index.
Legacy mode returns `(state, carry, (tp, kp))`; raw-RK mode returns
`(state, carry, rates, (tp, kp))`. Enable it only for the cumulus consumer that
needs it. Eight CAM tests pass; erasing the diagnostics fails numerically.
The cu=0 program and all 61 constants remain exact, so TC15 is reused. No new
carry fields or second PBL call are introduced. The two requested Swiss surface
outputs total 14,112 bytes [inferred]; actual GPU cost awaits measurement.
