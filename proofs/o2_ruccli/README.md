# RUC CLI initialization and history qualification

The CLI accepts `sf_surface_physics=3` with Thompson (`mp_physics=8`), MYNN's
surface layer (`sf_sfclay_physics=5`), and real.exe-prepared RUC wrfinput using
`num_soil_layers=6` or `9`. The model loader checks the existing RUC soil depths
and dimensions. Soil interpolation belongs to WRF real.exe preprocessing;
Noah four-layer input must be prepared again for RUC.

Both daily and nested loaders bind the RUC static and land bundles. The daily
path retains the full carry and global clock across hourly and auxhist segments.
The surface hook calls SFCDIAGS_RUCLSM immediately after LSMRUC, using updated
MAVAIL for CQS and the surface layer's CHS2/CQS2. Its T2/Q2 and surface diagnostics
remain in the existing history carry. The writer emits the evolving soil/snow
fields with the selected soil dimension. Both restart formats preserve RUC land,
held radiation, and held history. Radiation consumes step-start RUC albedo and
emissivity through array views of the retained land interface.

`build_oracle.py` compiles unmodified pristine WRF `module_sf_ruclsm.F` and
`module_sf_sfcdiags_ruclsm.F`, with WRF constants/error modules and monitor stubs.
`init_diag_savepoints.json` contains source hashes, compiler flags, and outputs
for both REAL=4 and REAL=8 builds. Twelve diagnostic columns cover warm/frozen
saturation, positive/negative fluxes, oversaturation, and low/threshold exchange
coefficients. Initialization covers warm/frozen soil, water and ice branches,
both land-use tables and both soil counts. The CLI refuses sea ice and lakes.
The existing 48-regime LSMRUC solver oracle remains the solver evidence.

CPU gates: 10 pristine parity checks, 23 admission/coupling/history/restart and
REAL-lowering checks, plus 170 shared consumers and 82 writer/daily/nested
regressions (four unavailable-reference skips). The final focused run's single
column-fixture correction was verified separately; completed unchanged oracle
checks were carried forward. Removing the final Q2 saturation bound causes two
numerical assertion failures, with no setup errors. The release fast-default
program remains exactly `2bf12cb8374d3a84a440e47dea4b77b0c7436108fc2ee2616bba6ec79b17962a`.

Changed-component CPU timing on core 25, 42×42 columns: compile plus execution
0.139 s, warm median 0.030 ms; REAL diagnostic HLO has no fp64. RUC's held
history adds six float32 planes (42,336 bytes here); other configurations add
zero planes and retain the identical release program. Output materializes RUC
land in one batched transfer at the output boundary. GPU kernel launches,
workspace, compile/readiness and coupled-forecast qualification are pending.

Run the focused tests with the current lane's CPU assignment:

```sh
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; \
JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 taskset -c 25 nice -n 19 \
  timeout 240 pytest -q tests/test_o2_ruccli.py \
  --basetemp <USER_HOME>/wrf_gpu2_lanes/o2-ruccli/pytest
```

Check the QUIET flag on the command line before starting. The release identity
wrapper reuses `proofs/o1_nlbind/probe.py` and changes only its scratch directory.
`cpu_smoke.py` exercises the real CLI loader and production domain-tree steps on
Swiss input. Its private soil profiles are interpolated for plumbing smoke;
pristine oracle checks supply initialization/diagnostic fidelity evidence.
Two production steps completed with every carry leaf finite and zero guard
events: load 151.7 s, first step including cold compilation 421.5 s, second
step 7.49 s (CPU, core 25, legacy defaults, census enabled). This is a functional
smoke result; GPU and real.exe-prepared coupled forecast qualification remain pending.
Receipts and logs live under `<USER_HOME>/wrf_gpu2_lanes/o2-ruccli/`.
