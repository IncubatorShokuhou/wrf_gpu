# Initial carry seed ownership

`OWNERSHIP_GATE.json` binds the source and CPU receipts for the RUC donation
failure. `_initial_carry_for_run` now copies only buffers shared with retained
namelist/static holders, once at initialization. Donation remains enabled.

The regression walks every carry leaf against all retained holder arrays by
object identity and buffer pointer, then executes actual CPU PjRt donation.
It covers RUC, Noah-classic, slab, Pleim-Xiu, Noah-MP, lake, grid/spectral FDDA,
and the post-initialization SLUCM and tslist seeds. All 14 tests pass; removing
the guard causes six ownership AssertionErrors with zero setup errors.

The actual Madrid RUC land-plus-XLAND capsule copies 479,808 bytes once; three
donations preserve all retained seed and output bytes. Its existing advance
guard subsequently copies zero leaves. CPU warm host-seam cost is 1.517 ms;
this capsule measurement is not full initialization, GPU timing or S1.

The 212 selected shared consumer checks pass. The real Swiss CLI release
jaxpr remains `2bf12cb8374d3a84a440e47dea4b77b0c7436108fc2ee2616bba6ec79b17962a`.
Source sealing proves all model AST unchanged except the new host helper and
the initializer's return, with no physics, donation policy or carry schema edit.
Existing numerical qualification is reused for this copy-only change. The
downstream RUC GPU forecast rerun is assigned to s1-gpuqual.

Reproduce the ownership tests with the repository's CPU/QUIET/core rules:

```bash
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 OMP_NUM_THREADS=1 \
  taskset -c 9 nice -n 19 timeout 240 python -m pytest \
  tests/test_initial_carry_seed_ownership.py -q \
  --basetemp <USER_HOME>/wrf_gpu2_lanes/s2-rucfix/pytest
```

Use core 13 during the 19:30Z–04:00Z nightly reservation. Evidence scripts,
logs, XML and JSON remain under `<USER_HOME>/wrf_gpu2_lanes/s2-rucfix/`;
measured commits are retained in the evidence refs recorded in the manifest.
